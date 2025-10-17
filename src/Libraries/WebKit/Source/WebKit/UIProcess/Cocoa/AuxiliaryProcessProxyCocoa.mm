/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 8, 2024.
 *
 * Licensed under the Apache License, Version 2.0 (the ""License"");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at:
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an ""AS IS"" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Please contact NeXTHub Corporation, 651 N Broad St, Suite 201, 
 * Middletown, DE 19709, New Castle County, USA.
 *
 */
#import "config.h"
#import "AuxiliaryProcessProxy.h"

#import "AuxiliaryProcessMessages.h"
#import "ProcessAssertion.h"
#import "XPCUtilities.h"
#import <WebCore/SharedBuffer.h>
#import <WebCore/WebMAudioUtilitiesCocoa.h>
#import <mach/mach_init.h>
#import <mach/task.h>
#import <mach/task_info.h>
#import <wtf/MonotonicTime.h>
#import <wtf/RunLoop.h>
#import <wtf/Scope.h>
#import <wtf/cocoa/VectorCocoa.h>

#import <pal/cf/AudioToolboxSoftLink.h>

namespace WebKit {

#if HAVE(AUDIO_COMPONENT_SERVER_REGISTRATIONS)
RefPtr<WebCore::SharedBuffer> AuxiliaryProcessProxy::fetchAudioComponentServerRegistrations()
{
    using namespace PAL;

    CFDataRef registrations { nullptr };

    if (!PAL::isAudioToolboxCoreFrameworkAvailable() || !PAL::canLoad_AudioToolboxCore_AudioComponentFetchServerRegistrations())
        return nullptr;
    
    WebCore::registerOpusDecoderIfNeeded();
    WebCore::registerVorbisDecoderIfNeeded();

    if (noErr != AudioComponentFetchServerRegistrations(&registrations) || !registrations)
        return nullptr;

    return WebCore::SharedBuffer::create(adoptCF(registrations).get());
}
#endif

Vector<String> AuxiliaryProcessProxy::platformOverrideLanguages() const
{
    static const NeverDestroyed<Vector<String>> overrideLanguages = makeVector<String>([[NSUserDefaults standardUserDefaults] stringArrayForKey:@"AppleLanguages"]);
    return overrideLanguages;
}

// This may be called from the AuxiliaryProcessProxy destructor, so do not pass the `this` pointer
// to other functions, or call any virtual functions.
void AuxiliaryProcessProxy::platformStartConnectionTerminationWatchdog()
{
#if USE(RUNNINGBOARD)
    if (m_startedTerminationWatchdog)
        return;

    m_startedTerminationWatchdog = true;

    // Deploy a watchdog in the UI process, since the child process may be suspended.
    // If 30s is insufficient for any outstanding activity to complete cleanly, then it will be killed.
    static constexpr ASCIILiteral reason = "XPCConnectionTerminationWatchdog"_s;

#if USE(EXTENSIONKIT_PROCESS_TERMINATION)
    auto maybeExtensionProcess = extensionProcess();
    if (!maybeExtensionProcess)
        return;

    Ref assertion = ProcessAndUIAssertion::create(processID(), reason, ProcessAssertionType::Background, environmentIdentifier(), extensionProcess());
    auto terminationHandler = [assertion = WTFMove(assertion), extensionProcess = WTFMove(*maybeExtensionProcess)] {
        extensionProcess.invalidate();
    };
#else
    if (!m_connection)
        return;

    OSObjectPtr xpcConnection = m_connection->xpcConnection();
    if (!xpcConnection)
        return;

    Ref assertion = ProcessAndUIAssertion::create(processID(), reason, ProcessAssertionType::Background, environmentIdentifier());
    auto terminationHandler = [assertion = WTFMove(assertion), xpcConnection = WTFMove(xpcConnection)]() {
        terminateWithReason(xpcConnection.get(), ReasonCode::WatchdogTimerFired, reason);
    };
#endif

    RunLoop::protectedMain()->dispatchAfter(30_s, WTFMove(terminationHandler));
#endif // USE(RUNNINGBOARD)
}

#if USE(EXTENSIONKIT)
std::optional<ExtensionProcess> AuxiliaryProcessProxy::extensionProcess() const
{
    if (!m_processLauncher)
        return std::nullopt;
    return m_processLauncher->extensionProcess();
}

LaunchGrant* AuxiliaryProcessProxy::launchGrant() const
{
    return m_processLauncher ? m_processLauncher->launchGrant() : nullptr;
}
#endif

std::optional<AuxiliaryProcessProxy::TaskInfo> AuxiliaryProcessProxy::taskInfo() const
{
    auto pid = processID();
    if (!pid)
        return std::nullopt;

    mach_port_t task = MACH_PORT_NULL;
    if (task_name_for_pid(mach_task_self(), pid, &task) != KERN_SUCCESS)
        return std::nullopt;

    auto scope = makeScopeExit([task]() {
        mach_port_deallocate(mach_task_self(), task);
    });

    mach_task_basic_info_data_t basicInfo;
    mach_msg_type_number_t basicInfoCount = MACH_TASK_BASIC_INFO_COUNT;

    if (task_info(task, MACH_TASK_BASIC_INFO, (task_info_t)&basicInfo, &basicInfoCount) != KERN_SUCCESS)
        return std::nullopt;

    task_absolutetime_info_data_t timeInfo;
    mach_msg_type_number_t timeInfoCount = TASK_ABSOLUTETIME_INFO_COUNT;

    if (task_info(task, TASK_ABSOLUTETIME_INFO, (task_info_t)&timeInfo, &timeInfoCount) != KERN_SUCCESS)
        return std::nullopt;

    task_vm_info_data_t vmInfo;
    mach_msg_type_number_t vmInfoCount = TASK_VM_INFO_REV1_COUNT;

    if (task_info(task, TASK_VM_INFO, (task_info_t)&vmInfo, &vmInfoCount) != KERN_SUCCESS)
        return std::nullopt;

    // ProcessThrottler's "suspend" state is a bit of a misnomer, because it could mean that the
    // process either is holding the "Suspended" assertion (in which case it's actually still not
    // task_suspended), or it is holding no assertions (in which case it's actually suspended).
    // Also, other processes (like NetworkProcess) can acquire assertions on the process that
    // UIProcess doesn't know about and which prevent the process from suspending.
    //
    // So we only tell the client that the task is suspended if we're sure that the task is actually
    // suspended (by consulting suspend_count).
    auto state = throttler().currentState();
    if (state == ProcessThrottleState::Suspended && !basicInfo.suspend_count)
        state = ProcessThrottleState::Background;

    return TaskInfo {
        pid,
        state,
        MonotonicTime::fromMachAbsoluteTime(timeInfo.total_user).secondsSinceEpoch(),
        MonotonicTime::fromMachAbsoluteTime(timeInfo.total_system).secondsSinceEpoch(),
        static_cast<size_t>(vmInfo.phys_footprint)
    };
}

#if ENABLE(CFPREFS_DIRECT_MODE)
void AuxiliaryProcessProxy::notifyPreferencesChanged(const String& domain, const String& key, const std::optional<String>& encodedValue)
{
    send(Messages::AuxiliaryProcess::PreferenceDidUpdate(domain, key, encodedValue), 0);
}
#endif

} // namespace WebKit
