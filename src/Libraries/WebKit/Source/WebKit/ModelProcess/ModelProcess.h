/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 21, 2022.
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
#pragma once

#if ENABLE(MODEL_PROCESS)

#include "AuxiliaryProcess.h"
#include "SandboxExtension.h"
#include "SharedPreferencesForWebProcess.h"
#include <WebCore/ProcessIdentifier.h>
#include <WebCore/Timer.h>
#include <pal/SessionID.h>
#include <wtf/Function.h>
#include <wtf/MemoryPressureHandler.h>
#include <wtf/MonotonicTime.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/WeakPtr.h>

#if PLATFORM(VISION) && ENABLE(GPU_PROCESS)
namespace IPC {
class SharedFileHandle;
}
#endif

namespace WebKit {

class ModelConnectionToWebProcess;
struct ModelProcessConnectionParameters;
struct ModelProcessCreationParameters;

class ModelProcess final : public AuxiliaryProcess, public ThreadSafeRefCounted<ModelProcess> {
    WTF_MAKE_NONCOPYABLE(ModelProcess);
    WTF_MAKE_TZONE_ALLOCATED(ModelProcess);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(ModelProcess);
public:
    explicit ModelProcess(AuxiliaryProcessInitializationParameters&&);
    ~ModelProcess();
    static constexpr WTF::AuxiliaryProcessType processType = WTF::AuxiliaryProcessType::Model;

    void ref() const final { ThreadSafeRefCounted::ref(); }
    void deref() const final { ThreadSafeRefCounted::deref(); }

    void removeModelConnectionToWebProcess(ModelConnectionToWebProcess&);

    void prepareToSuspend(bool isSuspensionImminent, MonotonicTime estimatedSuspendTime, CompletionHandler<void()>&&);
    void processDidResume();
    void resume();

    void connectionToWebProcessClosed(IPC::Connection&);

    ModelConnectionToWebProcess* webProcessConnection(WebCore::ProcessIdentifier) const;

    void tryExitIfUnusedAndUnderMemoryPressure();

    const String& applicationVisibleName() const { return m_applicationVisibleName; }

#if PLATFORM(VISION) && ENABLE(GPU_PROCESS)
    void requestSharedSimulationConnection(WebCore::ProcessIdentifier, CompletionHandler<void(std::optional<IPC::SharedFileHandle>)>&&);
#endif

    void webProcessConnectionCountForTesting(CompletionHandler<void(uint64_t)>&&);

private:
    void lowMemoryHandler(Critical, Synchronous);

    // AuxiliaryProcess
    void initializeProcess(const AuxiliaryProcessInitializationParameters&) override;
    void initializeProcessName(const AuxiliaryProcessInitializationParameters&) override;
    void initializeSandbox(const AuxiliaryProcessInitializationParameters&, SandboxInitializationParameters&) override;
    bool shouldTerminate() override;

    void tryExitIfUnused();
    bool canExitUnderMemoryPressure() const;

    // IPC::Connection::Client
    void didReceiveMessage(IPC::Connection&, IPC::Decoder&) override;

    // Message Handlers
    void initializeModelProcess(ModelProcessCreationParameters&&, CompletionHandler<void()>&&);
    void createModelConnectionToWebProcess(WebCore::ProcessIdentifier, PAL::SessionID, IPC::Connection::Handle&&, ModelProcessConnectionParameters&&, CompletionHandler<void()>&&);
    void sharedPreferencesForWebProcessDidChange(WebCore::ProcessIdentifier, SharedPreferencesForWebProcess&&, CompletionHandler<void()>&&);
    void addSession(PAL::SessionID);
    void removeSession(PAL::SessionID);

#if ENABLE(CFPREFS_DIRECT_MODE)
    void dispatchSimulatedNotificationsForPreferenceChange(const String& key) final;
#endif

    // Connections to WebProcesses.
    HashMap<WebCore::ProcessIdentifier, Ref<ModelConnectionToWebProcess>> m_webProcessConnections;
    MonotonicTime m_creationTime { MonotonicTime::now() };

    HashSet<PAL::SessionID> m_sessions;

    WebCore::Timer m_idleExitTimer;
    String m_applicationVisibleName;
};

} // namespace WebKit

#endif // ENABLE(MODEL_PROCESS)
