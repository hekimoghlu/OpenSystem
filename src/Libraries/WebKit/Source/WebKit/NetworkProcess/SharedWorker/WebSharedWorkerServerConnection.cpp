/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 19, 2024.
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
#include "WebSharedWorkerServerConnection.h"

#include "Logging.h"
#include "MessageSenderInlines.h"
#include "NetworkConnectionToWebProcess.h"
#include "NetworkProcess.h"
#include "NetworkProcessProxyMessages.h"
#include "NetworkSession.h"
#include "WebSharedWorker.h"
#include "WebSharedWorkerObjectConnectionMessages.h"
#include "WebSharedWorkerServer.h"
#include <WebCore/WorkerFetchResult.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebKit {

#define MESSAGE_CHECK(assertion) MESSAGE_CHECK_BASE(assertion, m_contentConnection.get())

#define CONNECTION_RELEASE_LOG(fmt, ...) RELEASE_LOG(SharedWorker, "%p - [webProcessIdentifier=%" PRIu64 "] WebSharedWorkerServerConnection::" fmt, this, m_webProcessIdentifier.toUInt64(), ##__VA_ARGS__)
#define CONNECTION_RELEASE_LOG_ERROR(fmt, ...) RELEASE_LOG_ERROR(SharedWorker, "%p - [webProcessIdentifier=%" PRIu64 "] WebSharedWorkerServerConnection::" fmt, this, m_webProcessIdentifier.toUInt64(), ##__VA_ARGS__)

WTF_MAKE_TZONE_ALLOCATED_IMPL(WebSharedWorkerServerConnection);

Ref<WebSharedWorkerServerConnection> WebSharedWorkerServerConnection::create(NetworkProcess& networkProcess, WebSharedWorkerServer& server, IPC::Connection& connection, WebCore::ProcessIdentifier webProcessIdentifier)
{
    return adoptRef(*new WebSharedWorkerServerConnection(networkProcess, server, connection, webProcessIdentifier));
}

WebSharedWorkerServerConnection::WebSharedWorkerServerConnection(NetworkProcess& networkProcess, WebSharedWorkerServer& server, IPC::Connection& connection, WebCore::ProcessIdentifier webProcessIdentifier)
    : m_contentConnection(connection)
    , m_networkProcess(networkProcess)
    , m_server(server)
    , m_webProcessIdentifier(webProcessIdentifier)
{
    CONNECTION_RELEASE_LOG("WebSharedWorkerServerConnection:");
}

WebSharedWorkerServerConnection::~WebSharedWorkerServerConnection()
{
    CONNECTION_RELEASE_LOG("~WebSharedWorkerServerConnection:");
}

Ref<NetworkProcess> WebSharedWorkerServerConnection::protectedNetworkProcess()
{
    return m_networkProcess;
}

WebSharedWorkerServer* WebSharedWorkerServerConnection::server()
{
    return m_server.get();
}

const WebSharedWorkerServer* WebSharedWorkerServerConnection::server() const
{
    return m_server.get();
}

IPC::Connection* WebSharedWorkerServerConnection::messageSenderConnection() const
{
    return m_contentConnection.ptr();
}

NetworkSession* WebSharedWorkerServerConnection::session()
{
    CheckedPtr server = m_server.get();
    if (!server)
        return nullptr;
    return protectedNetworkProcess()->networkSession(server->sessionID());
}

void WebSharedWorkerServerConnection::requestSharedWorker(WebCore::SharedWorkerKey&& sharedWorkerKey, WebCore::SharedWorkerObjectIdentifier sharedWorkerObjectIdentifier, WebCore::TransferredMessagePort&& port, WebCore::WorkerOptions&& workerOptions)
{
    MESSAGE_CHECK(protectedNetworkProcess()->allowsFirstPartyForCookies(m_webProcessIdentifier, WebCore::RegistrableDomain::uncheckedCreateFromHost(sharedWorkerKey.origin.topOrigin.host())) != NetworkProcess::AllowCookieAccess::Terminate);
    MESSAGE_CHECK(sharedWorkerObjectIdentifier.processIdentifier() == m_webProcessIdentifier);
    MESSAGE_CHECK(sharedWorkerKey.name == workerOptions.name);
    CONNECTION_RELEASE_LOG("requestSharedWorker: sharedWorkerObjectIdentifier=%" PUBLIC_LOG_STRING, sharedWorkerObjectIdentifier.toString().utf8().data());
    if (auto* session = this->session())
        session->ensureSharedWorkerServer().requestSharedWorker(WTFMove(sharedWorkerKey), sharedWorkerObjectIdentifier, WTFMove(port), WTFMove(workerOptions));
}

void WebSharedWorkerServerConnection::sharedWorkerObjectIsGoingAway(WebCore::SharedWorkerKey&& sharedWorkerKey, WebCore::SharedWorkerObjectIdentifier sharedWorkerObjectIdentifier)
{
    MESSAGE_CHECK(sharedWorkerObjectIdentifier.processIdentifier() == m_webProcessIdentifier);
    CONNECTION_RELEASE_LOG("sharedWorkerObjectIsGoingAway: sharedWorkerObjectIdentifier=%" PUBLIC_LOG_STRING, sharedWorkerObjectIdentifier.toString().utf8().data());
    if (auto* session = this->session())
        session->ensureSharedWorkerServer().sharedWorkerObjectIsGoingAway(sharedWorkerKey, sharedWorkerObjectIdentifier);
}

void WebSharedWorkerServerConnection::suspendForBackForwardCache(WebCore::SharedWorkerKey&& sharedWorkerKey, WebCore::SharedWorkerObjectIdentifier sharedWorkerObjectIdentifier)
{
    MESSAGE_CHECK(sharedWorkerObjectIdentifier.processIdentifier() == m_webProcessIdentifier);
    CONNECTION_RELEASE_LOG("suspendForBackForwardCache: sharedWorkerObjectIdentifier=%" PUBLIC_LOG_STRING, sharedWorkerObjectIdentifier.toString().utf8().data());
    if (auto* session = this->session())
        session->ensureSharedWorkerServer().suspendForBackForwardCache(sharedWorkerKey, sharedWorkerObjectIdentifier);
}

void WebSharedWorkerServerConnection::resumeForBackForwardCache(WebCore::SharedWorkerKey&& sharedWorkerKey, WebCore::SharedWorkerObjectIdentifier sharedWorkerObjectIdentifier)
{
    MESSAGE_CHECK(sharedWorkerObjectIdentifier.processIdentifier() == m_webProcessIdentifier);
    CONNECTION_RELEASE_LOG("resumeForBackForwardCache: sharedWorkerObjectIdentifier=%" PUBLIC_LOG_STRING, sharedWorkerObjectIdentifier.toString().utf8().data());
    if (auto* session = this->session())
        session->ensureSharedWorkerServer().resumeForBackForwardCache(sharedWorkerKey, sharedWorkerObjectIdentifier);
}

void WebSharedWorkerServerConnection::fetchScriptInClient(const WebSharedWorker& sharedWorker, WebCore::SharedWorkerObjectIdentifier sharedWorkerObjectIdentifier, CompletionHandler<void(WebCore::WorkerFetchResult&&, WebCore::WorkerInitializationData&&)>&& completionHandler)
{
    CONNECTION_RELEASE_LOG("fetchScriptInClient: sharedWorkerObjectIdentifier=%" PUBLIC_LOG_STRING, sharedWorkerObjectIdentifier.toString().utf8().data());
    sendWithAsyncReply(Messages::WebSharedWorkerObjectConnection::FetchScriptInClient { sharedWorker.url(), sharedWorkerObjectIdentifier, sharedWorker.workerOptions() }, WTFMove(completionHandler));
}

void WebSharedWorkerServerConnection::notifyWorkerObjectOfLoadCompletion(WebCore::SharedWorkerObjectIdentifier sharedWorkerObjectIdentifier, const WebCore::ResourceError& error)
{
    CONNECTION_RELEASE_LOG("notifyWorkerObjectOfLoadCompletion: sharedWorkerObjectIdentifier=%" PUBLIC_LOG_STRING ", success=%d", sharedWorkerObjectIdentifier.toString().utf8().data(), error.isNull());
    send(Messages::WebSharedWorkerObjectConnection::NotifyWorkerObjectOfLoadCompletion { sharedWorkerObjectIdentifier, error });
}

void WebSharedWorkerServerConnection::postErrorToWorkerObject(WebCore::SharedWorkerObjectIdentifier sharedWorkerObjectIdentifier, const String& errorMessage, int lineNumber, int columnNumber, const String& sourceURL, bool isErrorEvent)
{
    CONNECTION_RELEASE_LOG_ERROR("postErrorToWorkerObject: sharedWorkerObjectIdentifier=%" PUBLIC_LOG_STRING, sharedWorkerObjectIdentifier.toString().utf8().data());
    send(Messages::WebSharedWorkerObjectConnection::PostErrorToWorkerObject { sharedWorkerObjectIdentifier, errorMessage, lineNumber, columnNumber, sourceURL, isErrorEvent });
}

#undef CONNECTION_RELEASE_LOG
#undef CONNECTION_RELEASE_LOG_ERROR
#undef MESSAGE_CHECK

} // namespace WebKit
