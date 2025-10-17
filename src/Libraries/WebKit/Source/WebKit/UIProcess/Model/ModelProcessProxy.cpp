/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 27, 2025.
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
#include "ModelProcessProxy.h"

#if ENABLE(MODEL_PROCESS)

#include "DrawingAreaProxy.h"
#include "Logging.h"
#include "ModelProcessConnectionParameters.h"
#include "ModelProcessCreationParameters.h"
#include "ModelProcessMessages.h"
#include "ModelProcessProxyMessages.h"
#include "ProcessTerminationReason.h"
#include "ProvisionalPageProxy.h"
#include "WebPageGroup.h"
#include "WebPageMessages.h"
#include "WebPageProxy.h"
#include "WebPreferences.h"
#include "WebProcessMessages.h"
#include "WebProcessPool.h"
#include "WebProcessProxy.h"
#include "WebProcessProxyMessages.h"
#include <WebCore/LogInitialization.h>
#include <wtf/CompletionHandler.h>
#include <wtf/LogInitialization.h>
#include <wtf/MachSendRight.h>
#include <wtf/RuntimeApplicationChecks.h>
#include <wtf/TZoneMallocInlines.h>

#define MESSAGE_CHECK(assertion) MESSAGE_CHECK_BASE(assertion, connection())

namespace WebKit {
using namespace WebCore;

static WeakPtr<ModelProcessProxy>& singleton()
{
    static NeverDestroyed<WeakPtr<ModelProcessProxy>> singleton;
    return singleton;
}

WTF_MAKE_TZONE_ALLOCATED_IMPL(ModelProcessProxy);

Ref<ModelProcessProxy> ModelProcessProxy::getOrCreate()
{
    ASSERT(RunLoop::isMain());
    if (auto& existingModelProcess = singleton()) {
        ASSERT(existingModelProcess->state() != State::Terminated);
        return *existingModelProcess;
    }
    Ref modelProcess = adoptRef(*new ModelProcessProxy);
    singleton() = modelProcess;
    return modelProcess;
}

ModelProcessProxy* ModelProcessProxy::singletonIfCreated()
{
    return singleton().get();
}

ModelProcessProxy::ModelProcessProxy()
    : AuxiliaryProcessProxy(WebProcessPool::anyProcessPoolNeedsUIBackgroundAssertion() ? ShouldTakeUIBackgroundAssertion::Yes : ShouldTakeUIBackgroundAssertion::No)
{
    connect();

    ModelProcessCreationParameters parameters;
    parameters.auxiliaryProcessParameters = auxiliaryProcessParameters();
    parameters.parentPID = getCurrentProcessID();

    // Initialize the model process.
    sendWithAsyncReply(Messages::ModelProcess::InitializeModelProcess(WTFMove(parameters)), [initializationActivityAndGrant = initializationActivityAndGrant()] () { }, 0);

    updateProcessAssertion();
}

ModelProcessProxy::~ModelProcessProxy() = default;

void ModelProcessProxy::terminateWebProcess(WebCore::ProcessIdentifier webProcessIdentifier)
{
    if (auto process = WebProcessProxy::processForIdentifier(webProcessIdentifier)) {
        MESSAGE_CHECK(process->sharedPreferencesForWebProcessValue().modelElementEnabled);
        MESSAGE_CHECK(process->sharedPreferencesForWebProcessValue().modelProcessEnabled);
        process->requestTermination(ProcessTerminationReason::RequestedByModelProcess);
    }
}

void ModelProcessProxy::getLaunchOptions(ProcessLauncher::LaunchOptions& launchOptions)
{
    launchOptions.processType = ProcessLauncher::ProcessType::Model;
    AuxiliaryProcessProxy::getLaunchOptions(launchOptions);
}

void ModelProcessProxy::connectionWillOpen(IPC::Connection&)
{
}

void ModelProcessProxy::processWillShutDown(IPC::Connection& connection)
{
    ASSERT_UNUSED(connection, &this->connection() == &connection);

#if PLATFORM(VISION) && ENABLE(GPU_PROCESS)
    m_didInitializeSharedSimulationConnection = false;
#endif
}

void ModelProcessProxy::createModelProcessConnection(WebProcessProxy& webProcessProxy, IPC::Connection::Handle&& connectionIdentifier, ModelProcessConnectionParameters&& parameters)
{
    if (auto* store = webProcessProxy.websiteDataStore())
        addSession(*store);

    RELEASE_LOG(ProcessSuspension, "%p - ModelProcessProxy is taking a background assertion because a web process is requesting a connection", this);
    startResponsivenessTimer(UseLazyStop::No);
    sendWithAsyncReply(Messages::ModelProcess::CreateModelConnectionToWebProcess { webProcessProxy.coreProcessIdentifier(), webProcessProxy.sessionID(), WTFMove(connectionIdentifier), WTFMove(parameters) }, [this, weakThis = WeakPtr { *this }]() mutable {
        if (!weakThis)
            return;
        stopResponsivenessTimer();
    }, 0, IPC::SendOption::DispatchMessageEvenWhenWaitingForSyncReply);
}

void ModelProcessProxy::sharedPreferencesForWebProcessDidChange(WebProcessProxy& webProcessProxy, SharedPreferencesForWebProcess&& sharedPreferencesForWebProcess, CompletionHandler<void()>&& completionHandler)
{
    sendWithAsyncReply(Messages::ModelProcess::SharedPreferencesForWebProcessDidChange { webProcessProxy.coreProcessIdentifier(), WTFMove(sharedPreferencesForWebProcess) }, WTFMove(completionHandler));
}

void ModelProcessProxy::modelProcessExited(ProcessTerminationReason reason)
{
    Ref protectedThis { *this };

    switch (reason) {
    case ProcessTerminationReason::ExceededMemoryLimit:
    case ProcessTerminationReason::ExceededCPULimit:
    case ProcessTerminationReason::RequestedByClient:
    case ProcessTerminationReason::IdleExit:
    case ProcessTerminationReason::Unresponsive:
    case ProcessTerminationReason::Crash:
        RELEASE_LOG_ERROR(Process, "%p - ModelProcessProxy::modelProcessExited: reason=%{public}s", this, processTerminationReasonToString(reason).characters());
        break;
    case ProcessTerminationReason::ExceededProcessCountLimit:
    case ProcessTerminationReason::NavigationSwap:
    case ProcessTerminationReason::RequestedByNetworkProcess:
    case ProcessTerminationReason::RequestedByGPUProcess:
    case ProcessTerminationReason::RequestedByModelProcess:
    case ProcessTerminationReason::GPUProcessCrashedTooManyTimes:
    case ProcessTerminationReason::ModelProcessCrashedTooManyTimes:
    case ProcessTerminationReason::NonMainFrameWebContentProcessCrash:
        ASSERT_NOT_REACHED();
        break;
    }

    if (singleton() == this)
        singleton() = nullptr;

    for (auto& processPool : WebProcessPool::allProcessPools())
        processPool->modelProcessExited(processID(), reason);
}

void ModelProcessProxy::processIsReadyToExit()
{
    RELEASE_LOG(Process, "%p - ModelProcessProxy::processIsReadyToExit:", this);
    terminate();
    modelProcessExited(ProcessTerminationReason::IdleExit); // May cause |this| to get deleted.
}

void ModelProcessProxy::addSession(const WebsiteDataStore& store)
{
    if (!canSendMessage())
        return;

    if (m_sessionIDs.contains(store.sessionID()))
        return;

    send(Messages::ModelProcess::AddSession { store.sessionID() }, 0);
    m_sessionIDs.add(store.sessionID());
}

void ModelProcessProxy::removeSession(PAL::SessionID sessionID)
{
    if (!canSendMessage())
        return;

    if (m_sessionIDs.remove(sessionID))
        send(Messages::ModelProcess::RemoveSession { sessionID }, 0);
}

void ModelProcessProxy::terminateForTesting()
{
    processIsReadyToExit();
}

void ModelProcessProxy::webProcessConnectionCountForTesting(CompletionHandler<void(uint64_t)>&& completionHandler)
{
    sendWithAsyncReply(Messages::ModelProcess::WebProcessConnectionCountForTesting(), WTFMove(completionHandler));
}

void ModelProcessProxy::didClose(IPC::Connection&)
{
    RELEASE_LOG_ERROR(Process, "%p - ModelProcessProxy::didClose:", this);
    modelProcessExited(ProcessTerminationReason::Crash); // May cause |this| to get deleted.
}

void ModelProcessProxy::didReceiveInvalidMessage(IPC::Connection& connection, IPC::MessageName messageName, int32_t)
{
    logInvalidMessage(connection, messageName);

    WebProcessPool::didReceiveInvalidMessage(messageName);

    // Terminate the model process.
    terminate();

    // Since we've invalidated the connection we'll never get a IPC::Connection::Client::didClose
    // callback so we'll explicitly call it here instead.
    didClose(connection);
}

void ModelProcessProxy::didFinishLaunching(ProcessLauncher* launcher, IPC::Connection::Identifier&& connectionIdentifier)
{
    bool didTerminate = !connectionIdentifier;

    AuxiliaryProcessProxy::didFinishLaunching(launcher, WTFMove(connectionIdentifier));

    if (didTerminate) {
        modelProcessExited(ProcessTerminationReason::Crash);
        return;
    }

#if PLATFORM(COCOA)
    if (auto networkProcess = NetworkProcessProxy::defaultNetworkProcess())
        networkProcess->sendXPCEndpointToProcess(*this);
#endif

    beginResponsivenessChecks();

    for (auto& processPool : WebProcessPool::allProcessPools())
        processPool->modelProcessDidFinishLaunching(processID());
}

void ModelProcessProxy::updateProcessAssertion()
{
    bool hasAnyForegroundWebProcesses = false;
    bool hasAnyBackgroundWebProcesses = false;

    for (auto& processPool : WebProcessPool::allProcessPools()) {
        hasAnyForegroundWebProcesses |= processPool->hasForegroundWebProcesses();
        hasAnyBackgroundWebProcesses |= processPool->hasBackgroundWebProcesses();
    }

    if (hasAnyForegroundWebProcesses) {
        if (!ProcessThrottler::isValidForegroundActivity(m_activityFromWebProcesses.get()))
            m_activityFromWebProcesses = throttler().foregroundActivity("Model for foreground view(s)"_s);
        return;
    }
    if (hasAnyBackgroundWebProcesses) {
        if (!ProcessThrottler::isValidBackgroundActivity(m_activityFromWebProcesses.get()))
            m_activityFromWebProcesses = throttler().backgroundActivity("Model for background view(s)"_s);
        return;
    }

    // Use std::exchange() instead of a simple nullptr assignment to avoid re-entering this
    // function during the destructor of the ProcessThrottler activity, before setting
    // m_activityFromWebProcesses.
    std::exchange(m_activityFromWebProcesses, nullptr);
}

void ModelProcessProxy::sendPrepareToSuspend(IsSuspensionImminent isSuspensionImminent, double remainingRunTime, CompletionHandler<void()>&& completionHandler)
{
    sendWithAsyncReply(Messages::ModelProcess::PrepareToSuspend(isSuspensionImminent == IsSuspensionImminent::Yes, MonotonicTime::now() + Seconds(remainingRunTime)), WTFMove(completionHandler), 0, { }, ShouldStartProcessThrottlerActivity::No);
}

void ModelProcessProxy::sendProcessDidResume(ResumeReason)
{
    if (canSendMessage())
        send(Messages::ModelProcess::ProcessDidResume(), 0);
}

#if HAVE(VISIBILITY_PROPAGATION_VIEW)
void ModelProcessProxy::didCreateContextForVisibilityPropagation(WebPageProxyIdentifier webPageProxyID, WebCore::PageIdentifier pageID, LayerHostingContextID contextID)
{
    RELEASE_LOG(Process, "ModelProcessProxy::didCreateContextForVisibilityPropagation: webPageProxyID: %" PRIu64 ", pagePID: %" PRIu64 ", contextID: %u", webPageProxyID.toUInt64(), pageID.toUInt64(), contextID);
    auto page = WebProcessProxy::webPage(webPageProxyID);
    if (!page) {
        RELEASE_LOG(Process, "ModelProcessProxy::didCreateContextForVisibilityPropagation() No WebPageProxy with this identifier");
        return;
    }

    MESSAGE_CHECK(page->preferences().modelElementEnabled());
    MESSAGE_CHECK(page->preferences().modelProcessEnabled());

    if (page->webPageIDInMainFrameProcess() == pageID) {
        page->didCreateContextInModelProcessForVisibilityPropagation(contextID);
        return;
    }
    auto* provisionalPage = page->provisionalPageProxy();
    if (provisionalPage && provisionalPage->webPageID() == pageID) {
        provisionalPage->didCreateContextInModelProcessForVisibilityPropagation(contextID);
        return;
    }
    RELEASE_LOG(Process, "ModelProcessProxy::didCreateContextForVisibilityPropagation() There was a WebPageProxy for this identifier, but it had the wrong WebPage identifier.");
}
#endif

void ModelProcessProxy::didBecomeUnresponsive()
{
    RELEASE_LOG_ERROR(Process, "ModelProcessProxy::didBecomeUnresponsive: ModelProcess with PID %d became unresponsive, terminating it", processID());
    terminate();
    modelProcessExited(ProcessTerminationReason::Unresponsive);
}

} // namespace WebKit

#undef MESSAGE_CHECK

#endif // ENABLE(MODEL_PROCESS)
