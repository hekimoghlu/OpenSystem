/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 16, 2022.
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
#include "NetworkSocketChannel.h"

#include "MessageSenderInlines.h"
#include "NetworkConnectionToWebProcess.h"
#include "NetworkProcess.h"
#include "NetworkSession.h"
#include "WebSocketChannelMessages.h"
#include "WebSocketTask.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebKit {
using namespace WebCore;

WTF_MAKE_TZONE_ALLOCATED_IMPL(NetworkSocketChannel);

RefPtr<NetworkSocketChannel> NetworkSocketChannel::create(NetworkConnectionToWebProcess& connection, PAL::SessionID sessionID, const ResourceRequest& request, const String& protocol, WebSocketIdentifier identifier, WebPageProxyIdentifier webPageProxyID, std::optional<FrameIdentifier> frameID, std::optional<PageIdentifier> pageID, const WebCore::ClientOrigin& clientOrigin, bool hadMainFrameMainResourcePrivateRelayed, bool allowPrivacyProxy, OptionSet<AdvancedPrivacyProtections> advancedPrivacyProtections, WebCore::StoredCredentialsPolicy storedCredentialsPolicy)
{
    Ref result = adoptRef(*new NetworkSocketChannel(connection, connection.protectedNetworkProcess()->networkSession(sessionID), request, protocol, identifier, webPageProxyID, frameID, pageID, clientOrigin, hadMainFrameMainResourcePrivateRelayed, allowPrivacyProxy, advancedPrivacyProtections, storedCredentialsPolicy));
    if (!result->m_socket) {
        result->didClose(0, "Cannot create a web socket task"_s);
        return nullptr;
    }
    return result;
}

NetworkSocketChannel::NetworkSocketChannel(NetworkConnectionToWebProcess& connection, NetworkSession* session, const ResourceRequest& request, const String& protocol, WebSocketIdentifier identifier, WebPageProxyIdentifier webPageProxyID, std::optional<FrameIdentifier> frameID, std::optional<PageIdentifier> pageID, const WebCore::ClientOrigin& clientOrigin, bool hadMainFrameMainResourcePrivateRelayed, bool allowPrivacyProxy, OptionSet<AdvancedPrivacyProtections> advancedPrivacyProtections, WebCore::StoredCredentialsPolicy storedCredentialsPolicy)
    : m_connectionToWebProcess(connection)
    , m_identifier(identifier)
    , m_session(session)
    , m_errorTimer(*this, &NetworkSocketChannel::sendDelayedError)
    , m_webPageProxyID(webPageProxyID)
{
    relaxAdoptionRequirement();
    if (!m_session)
        return;

    m_socket = m_session->createWebSocketTask(webPageProxyID, frameID, pageID, *this, request, protocol, clientOrigin, hadMainFrameMainResourcePrivateRelayed, allowPrivacyProxy, advancedPrivacyProtections, storedCredentialsPolicy);
    if (CheckedPtr socket = m_socket.get()) {
#if PLATFORM(COCOA)
        m_session->addWebSocketTask(webPageProxyID, *socket);
#endif
        socket->resume();
    }
}

NetworkSocketChannel::~NetworkSocketChannel()
{
    if (CheckedPtr socket = m_socket.get()) {
#if PLATFORM(COCOA)
        if (RefPtr sessionSet = m_session ? socket->sessionSet() : nullptr)
            m_session->removeWebSocketTask(*sessionSet, *socket);
#endif
        socket->cancel();
    }
}

Ref<NetworkConnectionToWebProcess> NetworkSocketChannel::protectedConnectionToWebProcess()
{
    return m_connectionToWebProcess.get();
}

void NetworkSocketChannel::sendString(std::span<const uint8_t> message, CompletionHandler<void()>&& callback)
{
    checkedSocket()->sendString(message, WTFMove(callback));
}

void NetworkSocketChannel::sendData(std::span<const uint8_t> data, CompletionHandler<void()>&& callback)
{
    checkedSocket()->sendData(data, WTFMove(callback));
}

void NetworkSocketChannel::finishClosingIfPossible()
{
    if (m_state == State::Open) {
        m_state = State::Closing;
        return;
    }
    ASSERT(m_state == State::Closing);
    m_state = State::Closed;
    protectedConnectionToWebProcess()->removeSocketChannel(m_identifier);
}

void NetworkSocketChannel::close(int32_t code, const String& reason)
{
    checkedSocket()->close(code, reason);
    finishClosingIfPossible();
}

void NetworkSocketChannel::didConnect(const String& subprotocol, const String& extensions)
{
    send(Messages::WebSocketChannel::DidConnect { subprotocol, extensions });
}

void NetworkSocketChannel::didReceiveText(const String& text)
{
    send(Messages::WebSocketChannel::DidReceiveText { text });
}

void NetworkSocketChannel::didReceiveBinaryData(std::span<const uint8_t> data)
{
    send(Messages::WebSocketChannel::DidReceiveBinaryData { data });
}

void NetworkSocketChannel::didClose(unsigned short code, const String& reason)
{
    if (m_errorTimer.isActive()) {
        m_closeInfo = std::make_pair(code, reason);
        return;
    }
    send(Messages::WebSocketChannel::DidClose { code, reason });
    finishClosingIfPossible();
}

void NetworkSocketChannel::didReceiveMessageError(String&& errorMessage)
{
    m_errorMessage = WTFMove(errorMessage);
    m_errorTimer.startOneShot(NetworkProcess::randomClosedPortDelay());
}

void NetworkSocketChannel::sendDelayedError()
{
    send(Messages::WebSocketChannel::DidReceiveMessageError { m_errorMessage });
    if (m_closeInfo) {
        send(Messages::WebSocketChannel::DidClose { m_closeInfo->first, m_closeInfo->second });
        finishClosingIfPossible();
    }
}

void NetworkSocketChannel::didSendHandshakeRequest(ResourceRequest&& request)
{
    send(Messages::WebSocketChannel::DidSendHandshakeRequest { request });
}

void NetworkSocketChannel::didReceiveHandshakeResponse(ResourceResponse&& response)
{
    response.sanitizeHTTPHeaderFields(ResourceResponse::SanitizationType::CrossOriginSafe);
    send(Messages::WebSocketChannel::DidReceiveHandshakeResponse { response });
}

IPC::Connection* NetworkSocketChannel::messageSenderConnection() const
{
    return &m_connectionToWebProcess->connection();
}

NetworkSession* NetworkSocketChannel::session() const
{
    return m_session.get();
}

CheckedPtr<WebSocketTask> NetworkSocketChannel::checkedSocket()
{
    return m_socket.get();
}

} // namespace WebKit
