/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 15, 2025.
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
#include "config.h"
#include "ProcessAssertion.h"

#include "AuxiliaryProcessProxy.h"
#include "WKBase.h"
#include <wtf/RunLoop.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebKit {

ASCIILiteral processAssertionTypeDescription(ProcessAssertionType type)
{
    switch (type) {
    case ProcessAssertionType::NearSuspended:
        return "near-suspended"_s;
    case ProcessAssertionType::Background:
        return "background"_s;
    case ProcessAssertionType::UnboundedNetworking:
        return "unbounded-networking"_s;
    case ProcessAssertionType::Foreground:
        return "foreground"_s;
    case ProcessAssertionType::MediaPlayback:
        return "media-playback"_s;
    case ProcessAssertionType::FinishTaskCanSleep:
        return "finish-task-can-sleep"_s;
    case ProcessAssertionType::FinishTaskInterruptable:
        return "finish-task-interruptible"_s;
    case ProcessAssertionType::BoostedJetsam:
        return "boosted-jetsam"_s;
    }
    return "unknown"_s;
}

WTF_MAKE_TZONE_ALLOCATED_IMPL(ProcessAssertion);
WTF_MAKE_TZONE_ALLOCATED_IMPL(ProcessAndUIAssertion);

void ProcessAssertion::acquireAssertion(Mode mode, CompletionHandler<void()>&& acquisisionHandler)
{
    if (mode == Mode::Async)
        acquireAsync(WTFMove(acquisisionHandler));
    else {
        acquireSync();
        if (acquisisionHandler)
            acquisisionHandler();
    }
}

#if !USE(EXTENSIONKIT)

Ref<ProcessAssertion> ProcessAssertion::create(ProcessID processID, const String& reason, ProcessAssertionType type, Mode mode, const String& environmentIdentifier, CompletionHandler<void()>&& acquisisionHandler)
{
    auto assertion = adoptRef(*new ProcessAssertion(processID, reason, type, environmentIdentifier));
    assertion->acquireAssertion(mode, WTFMove(acquisisionHandler));
    return assertion;
}

Ref<ProcessAssertion> ProcessAssertion::create(AuxiliaryProcessProxy& process, const String& reason, ProcessAssertionType type, Mode mode, CompletionHandler<void()>&& acquisisionHandler)
{
    auto assertion = adoptRef(*new ProcessAssertion(process.processID(), reason, type, process.environmentIdentifier()));
    assertion->acquireAssertion(mode, WTFMove(acquisisionHandler));
    return assertion;
}

Ref<ProcessAndUIAssertion> ProcessAndUIAssertion::create(AuxiliaryProcessProxy& process, const String& reason, ProcessAssertionType type, Mode mode, CompletionHandler<void()>&& acquisisionHandler)
{
    auto assertion = adoptRef(*new ProcessAndUIAssertion(process.processID(), reason, type, process.environmentIdentifier()));
    assertion->acquireAssertion(mode, WTFMove(acquisisionHandler));
    return assertion;
}

Ref<ProcessAndUIAssertion> ProcessAndUIAssertion::create(ProcessID processID, const String& reason, ProcessAssertionType type, const String& environmentIdentifier, Mode mode, CompletionHandler<void()>&& acquisisionHandler)
{
    auto assertion = adoptRef(*new ProcessAndUIAssertion(processID, reason, type, environmentIdentifier));
    assertion->acquireAssertion(mode, WTFMove(acquisisionHandler));
    return assertion;
}

#else

Ref<ProcessAssertion> ProcessAssertion::create(ProcessID processID, const String& reason, ProcessAssertionType type, Mode mode, const String& environmentIdentifier, CompletionHandler<void()>&& acquisisionHandler)
{
    auto assertion = adoptRef(*new ProcessAssertion(processID, reason, type, environmentIdentifier, std::nullopt));
    assertion->acquireAssertion(mode, WTFMove(acquisisionHandler));
    return assertion;
}

Ref<ProcessAssertion> ProcessAssertion::create(AuxiliaryProcessProxy& process, const String& reason, ProcessAssertionType type, Mode mode, CompletionHandler<void()>&& acquisisionHandler)
{
    auto assertion = adoptRef(*new ProcessAssertion(process.processID(), reason, type, process.environmentIdentifier(), process.extensionProcess()));
    assertion->acquireAssertion(mode, WTFMove(acquisisionHandler));
    return assertion;
}

Ref<ProcessAndUIAssertion> ProcessAndUIAssertion::create(AuxiliaryProcessProxy& process, const String& reason, ProcessAssertionType type, Mode mode, CompletionHandler<void()>&& acquisisionHandler)
{
    auto assertion = adoptRef(*new ProcessAndUIAssertion(process.processID(), reason, type, process.environmentIdentifier(), process.extensionProcess()));
    assertion->acquireAssertion(mode, WTFMove(acquisisionHandler));
    return assertion;
}

Ref<ProcessAndUIAssertion> ProcessAndUIAssertion::create(ProcessID processID, const String& reason, ProcessAssertionType type, const String& environmentIdentifier, std::optional<ExtensionProcess>&& extensionProcess, Mode mode, CompletionHandler<void()>&& acquisisionHandler)
{
    auto assertion = adoptRef(*new ProcessAndUIAssertion(processID, reason, type, environmentIdentifier, WTFMove(extensionProcess)));
    assertion->acquireAssertion(mode, WTFMove(acquisisionHandler));
    return assertion;
}

#endif

#if !PLATFORM(COCOA) || !USE(RUNNINGBOARD)

ProcessAssertion::ProcessAssertion(ProcessID pid, const String& reason, ProcessAssertionType assertionType, const String&)
    : m_assertionType(assertionType)
    , m_pid(pid)
    , m_reason(reason)
{
}

ProcessAssertion::~ProcessAssertion() = default;

double ProcessAssertion::remainingRunTimeInSeconds(ProcessID)
{
    return 0;
}

bool ProcessAssertion::isValid() const
{
    return true;
}

void ProcessAssertion::acquireAsync(CompletionHandler<void()>&& completionHandler)
{
    if (completionHandler)
        RunLoop::main().dispatch(WTFMove(completionHandler));
}

void ProcessAssertion::acquireSync()
{
}

ProcessAndUIAssertion::ProcessAndUIAssertion(ProcessID pid, const String& reason, ProcessAssertionType assertionType, const String& environmentIdentifier)
    : ProcessAssertion(pid, reason, assertionType, environmentIdentifier)
{
}

ProcessAndUIAssertion::~ProcessAndUIAssertion() = default;

#endif

} // namespace WebKit

