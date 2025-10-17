/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 22, 2024.
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
#include "WebTransportSession.h"

#include "MessageSenderInlines.h"
#include "NetworkConnectionToWebProcessMessages.h"
#include "NetworkProcessConnection.h"
#include "NetworkTransportSessionMessages.h"
#include "WebProcess.h"
#include "WebTransportSendStreamSink.h"
#include <WebCore/Exception.h>
#include <WebCore/WebTransportBidirectionalStreamConstructionParameters.h>
#include <WebCore/WebTransportSessionClient.h>
#include <wtf/Ref.h>
#include <wtf/RunLoop.h>

namespace WebKit {

Ref<WebCore::WebTransportSessionPromise> WebTransportSession::initialize(Ref<IPC::Connection>&& connection, ThreadSafeWeakPtr<WebCore::WebTransportSessionClient>&& client, const URL& url, const WebPageProxyIdentifier& pageID, const WebCore::ClientOrigin& clientOrigin)
{
    ASSERT(RunLoop::isMain());
    return connection->sendWithPromisedReply(Messages::NetworkConnectionToWebProcess::InitializeWebTransportSession(url, pageID, clientOrigin))->whenSettled(RunLoop::main(), [connection, client = WTFMove(client)] (auto&& identifier) mutable {
        ASSERT(RunLoop::isMain());
        if (!identifier || !*identifier)
            return WebCore::WebTransportSessionPromise::createAndReject();
        return WebCore::WebTransportSessionPromise::createAndResolve(adoptRef(*new WebTransportSession(WTFMove(connection), WTFMove(client), **identifier)));
    });
}

WebTransportSession::WebTransportSession(Ref<IPC::Connection>&& connection, ThreadSafeWeakPtr<WebCore::WebTransportSessionClient>&& client, WebTransportSessionIdentifier identifier)
    : m_connection(WTFMove(connection))
    , m_client(WTFMove(client))
    , m_identifier(identifier)
{
    ASSERT(RunLoop::isMain());
    RELEASE_ASSERT(WebProcess::singleton().isWebTransportEnabled());
    WebProcess::singleton().addWebTransportSession(m_identifier, *this);
}

WebTransportSession::~WebTransportSession()
{
    ASSERT(RunLoop::isMain());
    WebProcess::singleton().removeWebTransportSession(m_identifier);
    m_connection->send(Messages::NetworkConnectionToWebProcess::DestroyWebTransportSession(m_identifier), 0);
}

IPC::Connection* WebTransportSession::messageSenderConnection() const
{
    return m_connection.ptr();
}

uint64_t WebTransportSession::messageSenderDestinationID() const
{
    return m_identifier.toUInt64();
}

void WebTransportSession::receiveDatagram(std::span<const uint8_t> datagram, bool withFin, std::optional<WebCore::Exception>&& exception)
{
    ASSERT(RunLoop::isMain());
    if (auto strongClient = m_client.get())
        strongClient->receiveDatagram(datagram, withFin, WTFMove(exception));
    else
        ASSERT_NOT_REACHED();
}

void WebTransportSession::receiveIncomingUnidirectionalStream(WebCore::WebTransportStreamIdentifier identifier)
{
    ASSERT(RunLoop::isMain());
    if (RefPtr strongClient = m_client.get())
        strongClient->receiveIncomingUnidirectionalStream(identifier);
    else
        ASSERT_NOT_REACHED();
}

void WebTransportSession::receiveBidirectionalStream(WebCore::WebTransportStreamIdentifier identifier)
{
    ASSERT(RunLoop::isMain());
    if (RefPtr strongClient = m_client.get()) {
        strongClient->receiveBidirectionalStream(WebCore::WebTransportBidirectionalStreamConstructionParameters {
            identifier,
            WebTransportSendStreamSink::create(*this, identifier)
        });
    } else
        ASSERT_NOT_REACHED();
}

void WebTransportSession::streamReceiveBytes(WebCore::WebTransportStreamIdentifier identifier, std::span<const uint8_t> bytes, bool withFin, std::optional<WebCore::Exception>&& exception)
{
    ASSERT(RunLoop::isMain());
    if (RefPtr strongClient = m_client.get())
        strongClient->streamReceiveBytes(identifier, bytes, withFin, WTFMove(exception));
    else
        ASSERT_NOT_REACHED();
}

Ref<WebCore::WebTransportSendPromise> WebTransportSession::sendDatagram(std::span<const uint8_t> datagram)
{
    return sendWithPromisedReply(Messages::NetworkTransportSession::SendDatagram(datagram))->whenSettled(RunLoop::main(), [] (auto&& exception) {
        ASSERT(RunLoop::isMain());
        if (!exception)
            return WebCore::WebTransportSendPromise::createAndReject();
        return WebCore::WebTransportSendPromise::createAndResolve(*exception);
    });
}

Ref<WebCore::WritableStreamPromise> WebTransportSession::createOutgoingUnidirectionalStream()
{
    return sendWithPromisedReply(Messages::NetworkTransportSession::CreateOutgoingUnidirectionalStream())->whenSettled(RunLoop::main(), [weakThis = ThreadSafeWeakPtr { *this }] (auto&& identifier) mutable {
        ASSERT(RunLoop::isMain());
        RefPtr strongThis = weakThis.get();
        if (!identifier || !*identifier || !strongThis)
            return WebCore::WritableStreamPromise::createAndReject();
        return WebCore::WritableStreamPromise::createAndResolve(WebTransportSendStreamSink::create(*strongThis, **identifier));
    });
}

Ref<WebCore::BidirectionalStreamPromise> WebTransportSession::createBidirectionalStream()
{
    return sendWithPromisedReply(Messages::NetworkTransportSession::CreateBidirectionalStream())->whenSettled(RunLoop::main(), [weakThis = ThreadSafeWeakPtr { *this }] (auto&& identifier) mutable {
        ASSERT(RunLoop::isMain());
        RefPtr strongThis = weakThis.get();
        if (!identifier || !*identifier || !strongThis)
            return WebCore::BidirectionalStreamPromise::createAndReject();
        return WebCore::BidirectionalStreamPromise::createAndResolve(WebCore::WebTransportBidirectionalStreamConstructionParameters {
            **identifier,
            WebTransportSendStreamSink::create(*strongThis, **identifier)
        });
    });
}

Ref<WebCore::WebTransportSendPromise> WebTransportSession::streamSendBytes(WebCore::WebTransportStreamIdentifier identifier, std::span<const uint8_t> bytes, bool withFin)
{
    return sendWithPromisedReply(Messages::NetworkTransportSession::StreamSendBytes(identifier, bytes, withFin))->whenSettled(RunLoop::main(), [] (auto&& exception) {
        if (!exception)
            return WebCore::WebTransportSendPromise::createAndReject();
        return WebCore::WebTransportSendPromise::createAndResolve(*exception);
    });
}

void WebTransportSession::terminate(WebCore::WebTransportSessionErrorCode code, CString&& reason)
{
    send(Messages::NetworkTransportSession::Terminate(code, WTFMove(reason)));
}

void WebTransportSession::networkProcessCrashed()
{
    ASSERT(RunLoop::isMain());
    if (RefPtr strongClient = m_client.get())
        strongClient->networkProcessCrashed();
}

void WebTransportSession::cancelReceiveStream(WebCore::WebTransportStreamIdentifier identifier, std::optional<WebCore::WebTransportStreamErrorCode> errorCode)
{
    send(Messages::NetworkTransportSession::CancelReceiveStream(identifier, errorCode));
}

void WebTransportSession::cancelSendStream(WebCore::WebTransportStreamIdentifier identifier, std::optional<WebCore::WebTransportStreamErrorCode> errorCode)
{
    send(Messages::NetworkTransportSession::CancelSendStream(identifier, errorCode));
}

void WebTransportSession::destroyStream(WebCore::WebTransportStreamIdentifier identifier, std::optional<WebCore::WebTransportStreamErrorCode> errorCode)
{
    send(Messages::NetworkTransportSession::DestroyStream(identifier, errorCode));
}
}
