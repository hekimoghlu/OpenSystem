/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 31, 2022.
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
#import "config.h"
#import "NetworkTransportStream.h"

#import "NetworkTransportSession.h"
#import <WebCore/Exception.h>
#import <WebCore/ExceptionCode.h>
#import <pal/spi/cocoa/NetworkSPI.h>
#import <wtf/BlockPtr.h>
#import <wtf/CompletionHandler.h>
#import <wtf/cocoa/SpanCocoa.h>
#import <wtf/cocoa/VectorCocoa.h>

#import <pal/cocoa/NetworkSoftLink.h>

namespace WebKit {

NetworkTransportStream::NetworkTransportStream(NetworkTransportSession& session, nw_connection_t connection, NetworkTransportStreamType streamType)
    : m_identifier(WebCore::WebTransportStreamIdentifier::generate())
    , m_session(session)
    , m_connection(connection)
    , m_streamType(streamType)
{
    ASSERT(m_connection);
    ASSERT(m_session);
    switch (m_streamType) {
    case NetworkTransportStreamType::Bidirectional:
        m_streamState = NetworkTransportStreamState::Ready;
        break;
    case NetworkTransportStreamType::IncomingUnidirectional:
        m_streamState = NetworkTransportStreamState::WriteClosed;
        break;
    case NetworkTransportStreamType::OutgoingUnidirectional:
        m_streamState = NetworkTransportStreamState::ReadClosed;
        break;
    }
    if (m_streamType != NetworkTransportStreamType::OutgoingUnidirectional)
        receiveLoop();
}

void NetworkTransportStream::sendBytes(std::span<const uint8_t> data, bool withFin, CompletionHandler<void(std::optional<WebCore::Exception>&&)>&& completionHandler)
{
    if (m_streamState == NetworkTransportStreamState::WriteClosed) {
        completionHandler(WebCore::Exception(WebCore::ExceptionCode::InvalidStateError));
        return;
    }
    nw_connection_send(m_connection.get(), makeDispatchData(Vector(data)).get(), NW_CONNECTION_DEFAULT_MESSAGE_CONTEXT, withFin, makeBlockPtr([weakThis = WeakPtr { *this }, withFin = withFin, completionHandler = WTFMove(completionHandler)] (nw_error_t error) mutable {
        RefPtr strongThis = weakThis.get();
        if (!strongThis)
            return;
        if (error) {
            if (nw_error_get_error_domain(error) == nw_error_domain_posix && nw_error_get_error_code(error) == ECANCELED)
                completionHandler(std::nullopt);
            else
                completionHandler(WebCore::Exception(WebCore::ExceptionCode::NetworkError));
            return;
        }

        completionHandler(std::nullopt);

        if (withFin) {
            switch (strongThis->m_streamState) {
            case NetworkTransportStreamState::Ready:
                strongThis->m_streamState = NetworkTransportStreamState::WriteClosed;
                break;
            case NetworkTransportStreamState::ReadClosed:
                strongThis->cancelSend(std::nullopt);
                break;
            case NetworkTransportStreamState::WriteClosed:
                RELEASE_ASSERT_NOT_REACHED();
            }
        }
    }).get());
}

void NetworkTransportStream::receiveLoop()
{
    RELEASE_ASSERT(m_streamState != NetworkTransportStreamState::ReadClosed);
    nw_connection_receive(m_connection.get(), 0, std::numeric_limits<uint32_t>::max(), makeBlockPtr([weakThis = WeakPtr { *this }] (dispatch_data_t content, nw_content_context_t, bool withFin, nw_error_t error) {
        RefPtr strongThis = weakThis.get();
        if (!strongThis)
            return;
        RefPtr session = strongThis->m_session.get();
        if (!session)
            return;
        if (error) {
            if (!(nw_error_get_error_domain(error) == nw_error_domain_posix && nw_error_get_error_code(error) == ECANCELED))
                session->streamReceiveBytes(strongThis->m_identifier, { }, false, WebCore::Exception(WebCore::ExceptionCode::NetworkError));
            return;
        }

        ASSERT(content || withFin);

        // FIXME: Not only is this an unnecessary string copy, but it's also something that should probably be in WTF or FragmentedSharedBuffer.
        auto vectorFromData = [](dispatch_data_t content) {
            Vector<uint8_t> request;
            if (content) {
                dispatch_data_apply_span(content, [&](std::span<const uint8_t> buffer) {
                    request.append(buffer);
                    return true;
                });
            }
            return request;
        };

        session->streamReceiveBytes(strongThis->m_identifier, vectorFromData(content).span(), withFin, std::nullopt);

        if (withFin) {
            switch (strongThis->m_streamState) {
            case NetworkTransportStreamState::Ready:
                strongThis->m_streamState = NetworkTransportStreamState::ReadClosed;
                break;
            case NetworkTransportStreamState::WriteClosed:
                strongThis->cancelReceive(std::nullopt);
                break;
            case NetworkTransportStreamState::ReadClosed:
                RELEASE_ASSERT_NOT_REACHED();
            }
        } else
            strongThis->receiveLoop();
    }).get());
}

void NetworkTransportStream::setErrorCodeForStream(std::optional<WebCore::WebTransportStreamErrorCode> errorCode)
{
    if (!errorCode)
        return;

    // FIXME: Implement once rdar://141886375 is available in OS builds.
}

void NetworkTransportStream::cancel(std::optional<WebCore::WebTransportStreamErrorCode> errorCode)
{
    setErrorCodeForStream(errorCode);
    nw_connection_cancel(m_connection.get());
}

void NetworkTransportStream::cancelReceive(std::optional<WebCore::WebTransportStreamErrorCode> errorCode)
{
    switch (m_streamState) {
    case NetworkTransportStreamState::Ready: {
        setErrorCodeForStream(errorCode);
        m_streamState = NetworkTransportStreamState::ReadClosed;
        // FIXME: Implement once rdar://141886375 is available in OS builds.
        break;
    }
    case NetworkTransportStreamState::WriteClosed: {
        RefPtr session = m_session.get();
        if (!session)
            return;
        session->destroyStream(m_identifier, errorCode);
        break;
    }
    case NetworkTransportStreamState::ReadClosed:
        RELEASE_ASSERT_NOT_REACHED();
    }
}

void NetworkTransportStream::cancelSend(std::optional<WebCore::WebTransportStreamErrorCode> errorCode)
{
    switch (m_streamState) {
    case NetworkTransportStreamState::Ready: {
        setErrorCodeForStream(errorCode);
        m_streamState = NetworkTransportStreamState::WriteClosed;
        // FIXME: Implement once rdar://141886375 is available in OS builds.
        break;
    }
    case NetworkTransportStreamState::ReadClosed: {
        RefPtr session = m_session.get();
        if (!session)
            return;
        session->destroyStream(m_identifier, errorCode);
        break;
    }
    case NetworkTransportStreamState::WriteClosed:
        RELEASE_ASSERT_NOT_REACHED();
    }
}
}
