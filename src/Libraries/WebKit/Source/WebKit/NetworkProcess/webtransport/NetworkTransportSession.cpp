/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 19, 2024.
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
#include "NetworkTransportSession.h"

#include "MessageSenderInlines.h"
#include "NetworkConnectionToWebProcess.h"
#include "NetworkTransportStream.h"
#include "WebCore/Exception.h"
#include "WebCore/ExceptionCode.h"
#include "WebTransportSessionMessages.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebKit {

WTF_MAKE_TZONE_ALLOCATED_IMPL(NetworkTransportSession);

#if !PLATFORM(COCOA)
void NetworkTransportSession::initialize(NetworkConnectionToWebProcess&, URL&&, WebKit::WebPageProxyIdentifier&&, WebCore::ClientOrigin&&, CompletionHandler<void(RefPtr<NetworkTransportSession>&&)>&& completionHandler)
{
    completionHandler(nullptr);
}
#endif

NetworkTransportSession::~NetworkTransportSession() = default;

IPC::Connection* NetworkTransportSession::messageSenderConnection() const
{
    return m_connectionToWebProcess ? &m_connectionToWebProcess->connection() : nullptr;
}

uint64_t NetworkTransportSession::messageSenderDestinationID() const
{
    return identifier().toUInt64();
}

#if !PLATFORM(COCOA)
void NetworkTransportSession::sendDatagram(std::span<const uint8_t>, CompletionHandler<void(std::optional<WebCore::Exception>&&)>&& completionHandler)
{
    completionHandler(std::nullopt);
}
#endif

void NetworkTransportSession::streamSendBytes(WebCore::WebTransportStreamIdentifier identifier, std::span<const uint8_t> bytes, bool withFin, CompletionHandler<void(std::optional<WebCore::Exception>&&)>&& completionHandler)
{
    if (RefPtr stream = m_streams.get(identifier))
        stream->sendBytes(bytes, withFin, WTFMove(completionHandler));
    else
        completionHandler(WebCore::Exception { WebCore::ExceptionCode::InvalidStateError });
}

#if !PLATFORM(COCOA)
void NetworkTransportSession::createOutgoingUnidirectionalStream(CompletionHandler<void(std::optional<WebCore::WebTransportStreamIdentifier>)>&& completionHandler)
{
    completionHandler(std::nullopt);
}

void NetworkTransportSession::createBidirectionalStream(CompletionHandler<void(std::optional<WebCore::WebTransportStreamIdentifier>)>&& completionHandler)
{
    completionHandler(std::nullopt);
}
#endif

#if !PLATFORM(COCOA)
void NetworkTransportSession::terminate(WebCore::WebTransportSessionErrorCode, CString&&)
{
}
#endif

void NetworkTransportSession::receiveDatagram(std::span<const uint8_t> datagram, bool withFin, std::optional<WebCore::Exception>&& exception)
{
    send(Messages::WebTransportSession::ReceiveDatagram(datagram, withFin, WTFMove(exception)));
}

void NetworkTransportSession::streamReceiveBytes(WebCore::WebTransportStreamIdentifier identifier, std::span<const uint8_t> bytes, bool withFin, std::optional<WebCore::Exception>&& exception)
{
    send(Messages::WebTransportSession::StreamReceiveBytes(identifier, bytes, withFin, WTFMove(exception)));
}

void NetworkTransportSession::receiveIncomingUnidirectionalStream(WebCore::WebTransportStreamIdentifier identifier)
{
    send(Messages::WebTransportSession::ReceiveIncomingUnidirectionalStream(identifier));
}

void NetworkTransportSession::receiveBidirectionalStream(WebCore::WebTransportStreamIdentifier identifier)
{
    send(Messages::WebTransportSession::ReceiveBidirectionalStream(identifier));
}

void NetworkTransportSession::cancelReceiveStream(WebCore::WebTransportStreamIdentifier identifier, std::optional<WebCore::WebTransportStreamErrorCode> errorCode)
{
    if (RefPtr stream = m_streams.get(identifier))
        stream->cancelReceive(errorCode);
    // Stream could have been destroyed gracefully when reads and writes were completed.
}

void NetworkTransportSession::cancelSendStream(WebCore::WebTransportStreamIdentifier identifier, std::optional<WebCore::WebTransportStreamErrorCode> errorCode)
{
    if (RefPtr stream = m_streams.get(identifier))
        stream->cancelSend(errorCode);
    // Stream could have been destroyed gracefully when reads and writes were completed.
}

void NetworkTransportSession::destroyStream(WebCore::WebTransportStreamIdentifier identifier, std::optional<WebCore::WebTransportStreamErrorCode> errorCode)
{
    if (RefPtr stream = m_streams.get(identifier)) {
        stream->cancel(errorCode);
        m_streams.remove(identifier);
    }
    // Stream could have been destroyed gracefully when reads and writes were completed.
}

std::optional<SharedPreferencesForWebProcess> NetworkTransportSession::sharedPreferencesForWebProcess() const
{
    if (auto connectionToWebProcess = m_connectionToWebProcess.get())
        return connectionToWebProcess->sharedPreferencesForWebProcess();

    return std::nullopt;
}

}
