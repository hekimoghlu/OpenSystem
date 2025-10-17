/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 16, 2023.
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
#include "WorkerWebTransportSession.h"

#include "ScriptExecutionContext.h"
#include "WebTransport.h"
#include "WebTransportBidirectionalStreamConstructionParameters.h"
#include "WritableStreamSink.h"

namespace WebCore {

Ref<WorkerWebTransportSession> WorkerWebTransportSession::create(ScriptExecutionContextIdentifier contextID, WebTransportSessionClient& client)
{
    ASSERT(!RunLoop::isMain());
    return adoptRef(*new WorkerWebTransportSession(contextID, client));
}

WorkerWebTransportSession::~WorkerWebTransportSession() = default;

WorkerWebTransportSession::WorkerWebTransportSession(ScriptExecutionContextIdentifier contextID, WebTransportSessionClient& client)
    : m_contextID(contextID)
    , m_client(client)
{
    ASSERT(!RunLoop::isMain());
}

void WorkerWebTransportSession::attachSession(Ref<WebTransportSession>&& session)
{
    ASSERT(!m_session);
    m_session = WTFMove(session);
}

void WorkerWebTransportSession::receiveDatagram(std::span<const uint8_t> span, bool withFin, std::optional<Exception>&& exception)
{
    ASSERT(RunLoop::isMain());
    ScriptExecutionContext::postTaskTo(m_contextID, [vector = Vector<uint8_t> { span }, withFin, exception = WTFMove(exception), weakClient = m_client] (auto&) mutable {
        RefPtr client = weakClient.get();
        if (!client)
            return;
        client->receiveDatagram(vector.span(), withFin, WTFMove(exception));
    });
}

void WorkerWebTransportSession::networkProcessCrashed()
{
    ASSERT(RunLoop::isMain());
    ScriptExecutionContext::postTaskTo(m_contextID, [weakClient = m_client] (auto&) mutable {
        RefPtr client = weakClient.get();
        if (!client)
            return;
        client->networkProcessCrashed();
    });
}

void WorkerWebTransportSession::receiveIncomingUnidirectionalStream(WebTransportStreamIdentifier identifier)
{
    ASSERT(RunLoop::isMain());
    ScriptExecutionContext::postTaskTo(m_contextID, [identifier, weakClient = m_client] (auto&) mutable {
        RefPtr client = weakClient.get();
        if (!client)
            return;
        client->receiveIncomingUnidirectionalStream(identifier);
    });
}

void WorkerWebTransportSession::receiveBidirectionalStream(WebTransportBidirectionalStreamConstructionParameters&& parameters)
{
    ASSERT(RunLoop::isMain());
    ScriptExecutionContext::postTaskTo(m_contextID, [parameters = WTFMove(parameters), weakClient = m_client] (auto&) mutable {
        RefPtr client = weakClient.get();
        if (!client)
            return;
        client->receiveBidirectionalStream(WTFMove(parameters));
    });
}

void WorkerWebTransportSession::streamReceiveBytes(WebTransportStreamIdentifier identifier, std::span<const uint8_t> data, bool withFin, std::optional<Exception>&& exception)
{
    ASSERT(RunLoop::isMain());
    ScriptExecutionContext::postTaskTo(m_contextID, [identifier, data = Vector<uint8_t> { data }, withFin, exception = WTFMove(exception),  weakClient = m_client] (auto&) mutable {
        RefPtr client = weakClient.get();
        if (!client)
            return;
        client->streamReceiveBytes(identifier, data.span(), withFin, WTFMove(exception));
    });
}

Ref<WebTransportSendPromise> WorkerWebTransportSession::sendDatagram(std::span<const uint8_t> datagram)
{
    ASSERT(!RunLoop::isMain());
    if (RefPtr session = m_session)
        return session->sendDatagram(datagram);
    return WebTransportSendPromise::createAndReject();
}

Ref<WritableStreamPromise> WorkerWebTransportSession::createOutgoingUnidirectionalStream()
{
    ASSERT(!RunLoop::isMain());
    if (RefPtr session = m_session)
        return session->createOutgoingUnidirectionalStream();
    return WritableStreamPromise::createAndReject();
}

Ref<BidirectionalStreamPromise> WorkerWebTransportSession::createBidirectionalStream()
{
    ASSERT(!RunLoop::isMain());
    if (RefPtr session = m_session)
        return session->createBidirectionalStream();
    return BidirectionalStreamPromise::createAndReject();
}

void WorkerWebTransportSession::terminate(WebTransportSessionErrorCode code, CString&& reason)
{
    ASSERT(!RunLoop::isMain());
    if (RefPtr session = m_session)
        session->terminate(code, WTFMove(reason));
}

void WorkerWebTransportSession::cancelReceiveStream(WebTransportStreamIdentifier identifier, std::optional<WebTransportStreamErrorCode> errorCode)
{
    ASSERT(!RunLoop::isMain());
    if (RefPtr session = m_session)
        session->cancelReceiveStream(identifier, errorCode);
}

void WorkerWebTransportSession::cancelSendStream(WebTransportStreamIdentifier identifier, std::optional<WebTransportStreamErrorCode> errorCode)
{
    ASSERT(!RunLoop::isMain());
    if (RefPtr session = m_session)
        session->cancelSendStream(identifier, errorCode);
}

void WorkerWebTransportSession::destroyStream(WebTransportStreamIdentifier identifier, std::optional<WebTransportStreamErrorCode> errorCode)
{
    ASSERT(!RunLoop::isMain());
    if (RefPtr session = m_session)
        session->destroyStream(identifier, errorCode);
}

}
