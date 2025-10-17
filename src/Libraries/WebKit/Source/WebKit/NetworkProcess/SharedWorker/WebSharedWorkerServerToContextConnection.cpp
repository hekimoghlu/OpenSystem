/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 14, 2021.
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
#include "WebSharedWorkerServerToContextConnection.h"

#include "LoadedWebArchive.h"
#include "Logging.h"
#include "MessageSenderInlines.h"
#include "NetworkConnectionToWebProcess.h"
#include "NetworkProcess.h"
#include "NetworkProcessProxyMessages.h"
#include "RemoteWorkerType.h"
#include "WebSWServerConnection.h"
#include "WebSharedWorker.h"
#include "WebSharedWorkerContextManagerConnectionMessages.h"
#include "WebSharedWorkerServer.h"
#include <WebCore/ScriptExecutionContextIdentifier.h>
#include <wtf/MemoryPressureHandler.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebKit {

#define CONTEXT_CONNECTION_RELEASE_LOG(fmt, ...) RELEASE_LOG(SharedWorker, "%p - [webProcessIdentifier=%" PRIu64 "] WebSharedWorkerServerToContextConnection::" fmt, this, webProcessIdentifier() ? webProcessIdentifier()->toUInt64() : 0, ##__VA_ARGS__)

// We terminate the context connection after 5 seconds if it is no longer used by any SharedWorker objects,
// as a performance optimization.
constexpr Seconds idleTerminationDelay { 5_s };

WTF_MAKE_TZONE_ALLOCATED_IMPL(WebSharedWorkerServerToContextConnection);

Ref<WebSharedWorkerServerToContextConnection> WebSharedWorkerServerToContextConnection::create(NetworkConnectionToWebProcess& connection, const WebCore::Site& site, WebSharedWorkerServer& server)
{
    return adoptRef(*new WebSharedWorkerServerToContextConnection(connection, site, server));
}

WebSharedWorkerServerToContextConnection::WebSharedWorkerServerToContextConnection(NetworkConnectionToWebProcess& connection, const WebCore::Site& site, WebSharedWorkerServer& server)
    : m_connection(connection)
    , m_server(server)
    , m_site(site)
    , m_idleTerminationTimer(*this, &WebSharedWorkerServerToContextConnection::idleTerminationTimerFired)
{
    CONTEXT_CONNECTION_RELEASE_LOG("WebSharedWorkerServerToContextConnection:");
    relaxAdoptionRequirement();
    server.addContextConnection(*this);
}

WebSharedWorkerServerToContextConnection::~WebSharedWorkerServerToContextConnection()
{
    CONTEXT_CONNECTION_RELEASE_LOG("~WebSharedWorkerServerToContextConnection:");
    if (CheckedPtr server = m_server.get(); server && server->contextConnectionForRegistrableDomain(registrableDomain()) == this)
        server->removeContextConnection(*this);
}

std::optional<WebCore::ProcessIdentifier> WebSharedWorkerServerToContextConnection::webProcessIdentifier() const
{
    if (!m_connection)
        return std::nullopt;
    return m_connection->webProcessIdentifier();
}

IPC::Connection* WebSharedWorkerServerToContextConnection::ipcConnection() const
{
    return m_connection ? &m_connection->connection() : nullptr;
}

IPC::Connection* WebSharedWorkerServerToContextConnection::messageSenderConnection() const
{
    return ipcConnection();
}

uint64_t WebSharedWorkerServerToContextConnection::messageSenderDestinationID() const
{
    return 0;
}

void WebSharedWorkerServerToContextConnection::connectionIsNoLongerNeeded()
{
    CONTEXT_CONNECTION_RELEASE_LOG("connectionIsNoLongerNeeded:");
    if (RefPtr connection = m_connection.get())
        connection->sharedWorkerServerToContextConnectionIsNoLongerNeeded();
}

void WebSharedWorkerServerToContextConnection::postErrorToWorkerObject(WebCore::SharedWorkerIdentifier sharedWorkerIdentifier, const String& errorMessage, int lineNumber, int columnNumber, const String& sourceURL, bool isErrorEvent)
{
    CONTEXT_CONNECTION_RELEASE_LOG("postErrorToWorkerObject: sharedWorkerIdentifier=%" PRIu64, sharedWorkerIdentifier.toUInt64());
    if (CheckedPtr server = m_server.get())
        server->postErrorToWorkerObject(sharedWorkerIdentifier, errorMessage, lineNumber, columnNumber, sourceURL, isErrorEvent);
}

void WebSharedWorkerServerToContextConnection::sharedWorkerTerminated(WebCore::SharedWorkerIdentifier sharedWorkerIdentifier)
{
    CONTEXT_CONNECTION_RELEASE_LOG("sharedWorkerTerminated: sharedWorkerIdentifier=%" PRIu64, sharedWorkerIdentifier.toUInt64());
    if (CheckedPtr server = m_server.get())
        server->sharedWorkerTerminated(sharedWorkerIdentifier);
}

void WebSharedWorkerServerToContextConnection::launchSharedWorker(WebSharedWorker& sharedWorker)
{
    RefPtr connection = m_connection.get();
    if (!connection)
        return;

    connection->protectedNetworkProcess()->addAllowedFirstPartyForCookies(connection->webProcessIdentifier(), WebCore::RegistrableDomain::uncheckedCreateFromHost(sharedWorker.origin().topOrigin.host()), LoadedWebArchive::No, [] { });

    CONTEXT_CONNECTION_RELEASE_LOG("launchSharedWorker: sharedWorkerIdentifier=%" PRIu64, sharedWorker.identifier().toUInt64());
    sharedWorker.markAsRunning();
    auto initializationData = sharedWorker.initializationData();
    if (auto contextIdentifier = initializationData.clientIdentifier) {
        initializationData.clientIdentifier = WebCore::ScriptExecutionContextIdentifier { contextIdentifier->object(), *webProcessIdentifier() };
        if (RefPtr serviceWorkerOldConnection = connection->protectedNetworkProcess()->webProcessConnection(contextIdentifier->processIdentifier())) {
            if (RefPtr swOldConnection = serviceWorkerOldConnection->swConnection()) {
                if (auto clientData = swOldConnection->gatherClientData(*contextIdentifier)) {
                    swOldConnection->unregisterServiceWorkerClient(*contextIdentifier);
                    if (RefPtr swNewConnection = connection->swConnection()) {
                        clientData->serviceWorkerClientData.identifier = *initializationData.clientIdentifier;
                        swNewConnection->registerServiceWorkerClient(WTFMove(clientData->clientOrigin), WTFMove(clientData->serviceWorkerClientData), clientData->controllingServiceWorkerRegistrationIdentifier, WTFMove(clientData->userAgent));
                    }
                }
            }
        }
    }
    send(Messages::WebSharedWorkerContextManagerConnection::LaunchSharedWorker { sharedWorker.origin(), sharedWorker.identifier(), sharedWorker.workerOptions(), sharedWorker.fetchResult(), initializationData });
    sharedWorker.forEachSharedWorkerObject([protectedThis = Ref { *this }, sharedWorker = Ref { sharedWorker }](auto, auto& port) {
        protectedThis->postConnectEvent(sharedWorker, port, [](bool) { });
    });
}

void WebSharedWorkerServerToContextConnection::suspendSharedWorker(WebCore::SharedWorkerIdentifier identifier)
{
    send(Messages::WebSharedWorkerContextManagerConnection::SuspendSharedWorker { identifier });
}

void WebSharedWorkerServerToContextConnection::resumeSharedWorker(WebCore::SharedWorkerIdentifier identifier)
{
    send(Messages::WebSharedWorkerContextManagerConnection::ResumeSharedWorker { identifier });
}

void WebSharedWorkerServerToContextConnection::postConnectEvent(const WebSharedWorker& sharedWorker, const WebCore::TransferredMessagePort& port, CompletionHandler<void(bool)>&& completionHandler)
{
    CONTEXT_CONNECTION_RELEASE_LOG("postConnectEvent: sharedWorkerIdentifier=%" PRIu64, sharedWorker.identifier().toUInt64());
    sendWithAsyncReply(Messages::WebSharedWorkerContextManagerConnection::PostConnectEvent { sharedWorker.identifier(), port, sharedWorker.origin().clientOrigin.toString() }, WTFMove(completionHandler));
}

void WebSharedWorkerServerToContextConnection::terminateSharedWorker(const WebSharedWorker& sharedWorker)
{
    CONTEXT_CONNECTION_RELEASE_LOG("terminateSharedWorker: sharedWorkerIdentifier=%" PRIu64, sharedWorker.identifier().toUInt64());
    send(Messages::WebSharedWorkerContextManagerConnection::TerminateSharedWorker { sharedWorker.identifier() });
}

void WebSharedWorkerServerToContextConnection::addSharedWorkerObject(WebCore::SharedWorkerObjectIdentifier sharedWorkerObjectIdentifier)
{
    CONTEXT_CONNECTION_RELEASE_LOG("addSharedWorkerObject: sharedWorkerObjectIdentifier=%" PUBLIC_LOG_STRING, sharedWorkerObjectIdentifier.toString().utf8().data());
    auto& sharedWorkerObjects = m_sharedWorkerObjects.ensure(sharedWorkerObjectIdentifier.processIdentifier(), [] { return HashSet<WebCore::SharedWorkerObjectIdentifier> { }; }).iterator->value;
    ASSERT(!sharedWorkerObjects.contains(sharedWorkerObjectIdentifier));
    sharedWorkerObjects.add(sharedWorkerObjectIdentifier);

    if (webProcessIdentifier() != sharedWorkerObjectIdentifier.processIdentifier() && sharedWorkerObjects.size() == 1) {
        RefPtr connection = m_connection.get();
        auto identifier = webProcessIdentifier();
        if (connection && identifier)
            connection->protectedNetworkProcess()->send(Messages::NetworkProcessProxy::RegisterRemoteWorkerClientProcess { RemoteWorkerType::SharedWorker, sharedWorkerObjectIdentifier.processIdentifier(), *identifier }, 0);
    }

    m_idleTerminationTimer.stop();
}

void WebSharedWorkerServerToContextConnection::removeSharedWorkerObject(WebCore::SharedWorkerObjectIdentifier sharedWorkerObjectIdentifier)
{
    CONTEXT_CONNECTION_RELEASE_LOG("removeSharedWorkerObject: sharedWorkerObjectIdentifier=%" PUBLIC_LOG_STRING, sharedWorkerObjectIdentifier.toString().utf8().data());
    auto it = m_sharedWorkerObjects.find(sharedWorkerObjectIdentifier.processIdentifier());
    ASSERT(it != m_sharedWorkerObjects.end());
    if (it == m_sharedWorkerObjects.end())
        return;
    it->value.remove(sharedWorkerObjectIdentifier);
    if (!it->value.isEmpty())
        return;

    m_sharedWorkerObjects.remove(it);
    if (webProcessIdentifier() != sharedWorkerObjectIdentifier.processIdentifier()) {
        RefPtr connection = m_connection.get();
        auto identifier = webProcessIdentifier();
        if (connection && identifier)
            connection->protectedNetworkProcess()->send(Messages::NetworkProcessProxy::UnregisterRemoteWorkerClientProcess { RemoteWorkerType::SharedWorker, sharedWorkerObjectIdentifier.processIdentifier(), *identifier }, 0);
    }


    if (m_sharedWorkerObjects.isEmpty()) {
        CONTEXT_CONNECTION_RELEASE_LOG("removeSharedWorkerObject: connection is now idle, starting a timer to terminate it");
        ASSERT(!m_idleTerminationTimer.isActive());
        m_idleTerminationTimer.startOneShot(MemoryPressureHandler::singleton().isUnderMemoryPressure() || m_shouldTerminateWhenPossible ? 0_s : idleTerminationDelay);
    }
}

void WebSharedWorkerServerToContextConnection::idleTerminationTimerFired()
{
    RELEASE_ASSERT(m_sharedWorkerObjects.isEmpty());
    connectionIsNoLongerNeeded();
}

#undef CONTEXT_CONNECTION_RELEASE_LOG

} // namespace WebKit
