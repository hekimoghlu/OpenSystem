/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 11, 2024.
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

#include "AuxiliaryProcessProxy.h"
#include "ProcessLauncher.h"
#include "ProcessTerminationReason.h"
#include "ProcessThrottler.h"
#include "WebPageProxyIdentifier.h"
#include <WebCore/PageIdentifier.h>
#include <memory>
#include <pal/SessionID.h>
#include <wtf/CompletionHandler.h>
#include <wtf/TZoneMalloc.h>

#if HAVE(VISIBILITY_PROPAGATION_VIEW)
#include "LayerHostingContext.h"
#endif

#if PLATFORM(VISION) && ENABLE(GPU_PROCESS)
namespace IPC {
class SharedFileHandle;
}
#endif

namespace WebKit {

class WebProcessProxy;
class WebsiteDataStore;
struct ModelProcessConnectionParameters;
struct ModelProcessCreationParameters;
struct SharedPreferencesForWebProcess;

class ModelProcessProxy final : public AuxiliaryProcessProxy {
    WTF_MAKE_TZONE_ALLOCATED(ModelProcessProxy);
    WTF_MAKE_NONCOPYABLE(ModelProcessProxy);
    friend LazyNeverDestroyed<ModelProcessProxy>;
public:
    static Ref<ModelProcessProxy> getOrCreate();
    static ModelProcessProxy* singletonIfCreated();
    ~ModelProcessProxy();

    void createModelProcessConnection(WebProcessProxy&, IPC::Connection::Handle&& connectionIdentifier, ModelProcessConnectionParameters&&);
    void sharedPreferencesForWebProcessDidChange(WebProcessProxy&, SharedPreferencesForWebProcess&&, CompletionHandler<void()>&&);

    void updateProcessAssertion();

    void terminateForTesting();
    void webProcessConnectionCountForTesting(CompletionHandler<void(uint64_t)>&&);

    void removeSession(PAL::SessionID);

private:
    explicit ModelProcessProxy();

    void terminateWebProcess(WebCore::ProcessIdentifier);

    Type type() const final { return Type::Model; }

    void addSession(const WebsiteDataStore&);

    // AuxiliaryProcessProxy
    ASCIILiteral processName() const final { return "Model"_s; }

    void getLaunchOptions(ProcessLauncher::LaunchOptions&) override;
    void connectionWillOpen(IPC::Connection&) override;
    void processWillShutDown(IPC::Connection&) override;

    void modelProcessExited(ProcessTerminationReason);

    // ProcessThrottlerClient
    ASCIILiteral clientName() const final { return "ModelProcess"_s; }
    void sendPrepareToSuspend(IsSuspensionImminent, double remainingRunTime, CompletionHandler<void()>&&) final;
    void sendProcessDidResume(ResumeReason) final;

    // ProcessLauncher::Client
    void didFinishLaunching(ProcessLauncher*, IPC::Connection::Identifier&&) override;

    // IPC::Connection::Client
    void didReceiveMessage(IPC::Connection&, IPC::Decoder&) override;
    void didClose(IPC::Connection&) override;
    void didReceiveInvalidMessage(IPC::Connection&, IPC::MessageName, int32_t indexOfObjectFailingDecoding) override;

    // ResponsivenessTimer::Client
    void didBecomeUnresponsive() final;

    void processIsReadyToExit();

#if HAVE(VISIBILITY_PROPAGATION_VIEW)
    void didCreateContextForVisibilityPropagation(WebPageProxyIdentifier, WebCore::PageIdentifier, LayerHostingContextID);
#endif

#if PLATFORM(VISION) && ENABLE(GPU_PROCESS)
    void requestSharedSimulationConnection(WebCore::ProcessIdentifier, CompletionHandler<void(std::optional<IPC::SharedFileHandle>)>&&);
#endif

    ModelProcessCreationParameters processCreationParameters();

    RefPtr<ProcessThrottler::Activity> m_activityFromWebProcesses;

    HashSet<PAL::SessionID> m_sessionIDs;

#if PLATFORM(VISION) && ENABLE(GPU_PROCESS)
    bool m_didInitializeSharedSimulationConnection { false };
#endif
};

} // namespace WebKit

#endif // ENABLE(MODEL_PROCESS)
