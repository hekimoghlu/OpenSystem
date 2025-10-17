/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 7, 2024.
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
#include "ModelProcess.h"

#if ENABLE(MODEL_PROCESS)

#include "ArgumentCoders.h"
#include "Attachment.h"
#include "AuxiliaryProcessMessages.h"
#include "LogInitialization.h"
#include "ModelConnectionToWebProcess.h"
#include "ModelProcessConnectionParameters.h"
#include "ModelProcessCreationParameters.h"
#include "ModelProcessProxyMessages.h"
#include "WebPageProxyMessages.h"
#include "WebProcessPoolMessages.h"
#include <WebCore/CommonAtomStrings.h>
#include <WebCore/LogInitialization.h>
#include <WebCore/MemoryRelease.h>
#include <wtf/Algorithms.h>
#include <wtf/CallbackAggregator.h>
#include <wtf/LogInitialization.h>
#include <wtf/MemoryPressureHandler.h>
#include <wtf/OptionSet.h>
#include <wtf/RunLoop.h>
#include <wtf/Scope.h>
#include <wtf/TZoneMallocInlines.h>
#include <wtf/UniqueRef.h>
#include <wtf/text/AtomString.h>

#if PLATFORM(VISION) && ENABLE(GPU_PROCESS)
#include "SharedFileHandle.h"
#include <WebKitAdditions/WKREEngine.h>
#endif

namespace WebKit {

WTF_MAKE_TZONE_ALLOCATED_IMPL(ModelProcess);

// We wouldn't want the ModelProcess to repeatedly exit then relaunch when under memory pressure. In particular, we need to make sure the
// WebProcess has a chance to schedule work after the ModelProcess gets launched. For this reason, we make sure that the ModelProcess never
// idle-exits less than 5 seconds after getting launched. This amount of time should be sufficient for the WebProcess to schedule
// work in the ModelProcess.
constexpr Seconds minimumLifetimeBeforeIdleExit { 5_s };

ModelProcess::ModelProcess(AuxiliaryProcessInitializationParameters&& parameters)
    : m_idleExitTimer(*this, &ModelProcess::tryExitIfUnused)
{
    initialize(WTFMove(parameters));
    RELEASE_LOG(Process, "%p - ModelProcess::ModelProcess:", this);
}

ModelProcess::~ModelProcess() = default;

void ModelProcess::createModelConnectionToWebProcess(WebCore::ProcessIdentifier identifier, PAL::SessionID sessionID, IPC::Connection::Handle&& connectionHandle, ModelProcessConnectionParameters&& parameters, CompletionHandler<void()>&& completionHandler)
{
    RELEASE_LOG(Process, "%p - ModelProcess::createModelConnectionToWebProcess: processIdentifier=%" PRIu64, this, identifier.toUInt64());

    auto reply = makeScopeExit(WTFMove(completionHandler));
    // If sender exited before we received the identifier, the identifier
    // may not be valid.
    if (!connectionHandle)
        return;

#if PLATFORM(VISION) && ENABLE(GPU_PROCESS)
    WKREEngine::shared().initializeWithSharedSimulationConnectionGetterIfNeeded([identifier, weakThis = WeakPtr { *this }] (CompletionHandler<void(std::optional<IPC::SharedFileHandle>)>&& completionHandler) {
        RefPtr protectedThis = weakThis.get();
        if (!protectedThis) {
            completionHandler(std::nullopt);
            return;
        }

        protectedThis->requestSharedSimulationConnection(identifier, WTFMove(completionHandler));
    });
#endif

    auto newConnection = ModelConnectionToWebProcess::create(*this, identifier, sessionID, WTFMove(connectionHandle), WTFMove(parameters));

#if ENABLE(IPC_TESTING_API)
    if (parameters.ignoreInvalidMessageForTesting)
        newConnection->connection().setIgnoreInvalidMessageForTesting();
#endif

    ASSERT(!m_webProcessConnections.contains(identifier));
    m_webProcessConnections.add(identifier, WTFMove(newConnection));
}

void ModelProcess::sharedPreferencesForWebProcessDidChange(WebCore::ProcessIdentifier identifier, SharedPreferencesForWebProcess&& sharedPreferencesForWebProcess, CompletionHandler<void()>&& completionHandler)
{
    if (RefPtr connection = m_webProcessConnections.get(identifier))
        connection->updateSharedPreferencesForWebProcess(WTFMove(sharedPreferencesForWebProcess));
    completionHandler();
}

void ModelProcess::removeModelConnectionToWebProcess(ModelConnectionToWebProcess& connection)
{
    RELEASE_LOG(Process, "%p - ModelProcess::removeModelConnectionToWebProcess: processIdentifier=%" PRIu64, this, connection.webProcessIdentifier().toUInt64());
    ASSERT(m_webProcessConnections.contains(connection.webProcessIdentifier()));
    m_webProcessConnections.remove(connection.webProcessIdentifier());
    tryExitIfUnusedAndUnderMemoryPressure();
}

void ModelProcess::connectionToWebProcessClosed(IPC::Connection& connection)
{
}

bool ModelProcess::shouldTerminate()
{
    return m_webProcessConnections.isEmpty();
}

bool ModelProcess::canExitUnderMemoryPressure() const
{
    ASSERT(isMainRunLoop());
    for (auto& webProcessConnection : m_webProcessConnections.values()) {
        if (!webProcessConnection->allowsExitUnderMemoryPressure())
            return false;
    }
    return true;
}

void ModelProcess::tryExitIfUnusedAndUnderMemoryPressure()
{
    ASSERT(isMainRunLoop());
    if (!MemoryPressureHandler::singleton().isUnderMemoryPressure())
        return;

    tryExitIfUnused();
}

void ModelProcess::tryExitIfUnused()
{
    ASSERT(isMainRunLoop());
    if (!canExitUnderMemoryPressure()) {
        m_idleExitTimer.stop();
        return;
    }

    // To avoid exiting the ModelProcess too aggressively while under memory pressure and make sure the WebProcess gets a
    // change to schedule work, we don't exit if we've been running for less than |minimumLifetimeBeforeIdleExit|.
    // In case of simulated memory pressure, we ignore this rule to avoid flakiness in our benchmarks and tests.
    auto lifetime = MonotonicTime::now() - m_creationTime;
    if (lifetime < minimumLifetimeBeforeIdleExit && !MemoryPressureHandler::singleton().isSimulatingMemoryPressure()) {
        RELEASE_LOG(Process, "ModelProcess::tryExitIfUnused: ModelProcess is idle and under memory pressure but it is not exiting because it has just launched");
        // Check again after the process have lived long enough (minimumLifetimeBeforeIdleExit) to see if the ModelProcess
        // can idle-exit then.
        if (!m_idleExitTimer.isActive())
            m_idleExitTimer.startOneShot(minimumLifetimeBeforeIdleExit - lifetime);
        return;
    }
    m_idleExitTimer.stop();

    RELEASE_LOG(Process, "ModelProcess::tryExitIfUnused: ModelProcess is exiting because we are under memory pressure and the process is no longer useful.");
    parentProcessConnection()->send(Messages::ModelProcessProxy::ProcessIsReadyToExit(), 0);
}

void ModelProcess::lowMemoryHandler(Critical critical, Synchronous synchronous)
{
    RELEASE_LOG(Process, "ModelProcess::lowMemoryHandler: critical=%d, synchronous=%d", critical == Critical::Yes, synchronous == Synchronous::Yes);
    tryExitIfUnused();

    for (auto& connection : m_webProcessConnections.values())
        connection->lowMemoryHandler(critical, synchronous);

    WebCore::releaseGraphicsMemory(critical, synchronous);
}

void ModelProcess::initializeModelProcess(ModelProcessCreationParameters&& parameters, CompletionHandler<void()>&& completionHandler)
{
    CompletionHandlerCallingScope callCompletionHandler(WTFMove(completionHandler));

    applyProcessCreationParameters(parameters.auxiliaryProcessParameters);
    RELEASE_LOG(Process, "%p - ModelProcess::initializeModelProcess:", this);
    WTF::Thread::setCurrentThreadIsUserInitiated();
    WebCore::initializeCommonAtomStrings();

    auto& memoryPressureHandler = MemoryPressureHandler::singleton();
    memoryPressureHandler.setLowMemoryHandler([this] (Critical critical, Synchronous synchronous) {
        lowMemoryHandler(critical, synchronous);
    });
    memoryPressureHandler.install();

    m_applicationVisibleName = WTFMove(parameters.applicationVisibleName);

    // Match the QoS of the UIProcess since the model process is doing rendering on its behalf.
    WTF::Thread::setCurrentThreadIsUserInteractive(0);

    setLegacyPresentingApplicationPID(parameters.parentPID);

#if USE(OS_STATE)
    registerWithStateDumper("ModelProcess state"_s);
#endif
}

void ModelProcess::prepareToSuspend(bool isSuspensionImminent, MonotonicTime, CompletionHandler<void()>&& completionHandler)
{
    RELEASE_LOG(ProcessSuspension, "%p - ModelProcess::prepareToSuspend(), isSuspensionImminent: %d", this, isSuspensionImminent);

    lowMemoryHandler(Critical::Yes, Synchronous::Yes);
    completionHandler();
}

void ModelProcess::processDidResume()
{
    RELEASE_LOG(ProcessSuspension, "%p - ModelProcess::processDidResume()", this);
    resume();
}

void ModelProcess::resume()
{
}

ModelConnectionToWebProcess* ModelProcess::webProcessConnection(WebCore::ProcessIdentifier identifier) const
{
    return m_webProcessConnections.get(identifier);
}

#if PLATFORM(VISION) && ENABLE(GPU_PROCESS)
void ModelProcess::requestSharedSimulationConnection(WebCore::ProcessIdentifier webProcessIdentifier, CompletionHandler<void(std::optional<IPC::SharedFileHandle>)>&& completionHandler)
{
    parentProcessConnection()->sendWithAsyncReply(Messages::ModelProcessProxy::RequestSharedSimulationConnection(webProcessIdentifier), WTFMove(completionHandler));
}
#endif

void ModelProcess::webProcessConnectionCountForTesting(CompletionHandler<void(uint64_t)>&& completionHandler)
{
    completionHandler(ModelConnectionToWebProcess::objectCountForTesting());
}

void ModelProcess::addSession(PAL::SessionID sessionID)
{
    ASSERT(!m_sessions.contains(sessionID));
    m_sessions.add(sessionID);
}

void ModelProcess::removeSession(PAL::SessionID sessionID)
{
    ASSERT(m_sessions.contains(sessionID));
    m_sessions.remove(sessionID);
}

} // namespace WebKit

#endif // ENABLE(MODEL_PROCESS)
