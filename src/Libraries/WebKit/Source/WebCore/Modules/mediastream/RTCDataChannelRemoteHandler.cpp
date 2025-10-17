/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 19, 2025.
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
#include "RTCDataChannelRemoteHandler.h"

#if ENABLE(WEB_RTC)

#include "ProcessQualified.h"
#include "RTCDataChannelHandlerClient.h"
#include "RTCDataChannelRemoteHandlerConnection.h"
#include "ScriptExecutionContextIdentifier.h"
#include "SharedBuffer.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(RTCDataChannelRemoteHandler);

std::unique_ptr<RTCDataChannelRemoteHandler> RTCDataChannelRemoteHandler::create(RTCDataChannelIdentifier remoteIdentifier, RefPtr<RTCDataChannelRemoteHandlerConnection>&& connection)
{
    if (!connection)
        return nullptr;
    return makeUnique<RTCDataChannelRemoteHandler>(remoteIdentifier, connection.releaseNonNull());
}

RTCDataChannelRemoteHandler::RTCDataChannelRemoteHandler(RTCDataChannelIdentifier remoteIdentifier, Ref<RTCDataChannelRemoteHandlerConnection>&& connection)
    : m_remoteIdentifier(remoteIdentifier)
    , m_connection(WTFMove(connection))
{
}

RTCDataChannelRemoteHandler::~RTCDataChannelRemoteHandler() = default;

void RTCDataChannelRemoteHandler::didChangeReadyState(RTCDataChannelState state)
{
    m_client->didChangeReadyState(state);
}

void RTCDataChannelRemoteHandler::didReceiveStringData(String&& text)
{
    m_client->didReceiveStringData(text);
}

void RTCDataChannelRemoteHandler::didReceiveRawData(std::span<const uint8_t> data)
{
    m_client->didReceiveRawData(data);
}

void RTCDataChannelRemoteHandler::didDetectError(Ref<RTCError>&& error)
{
    m_client->didDetectError(WTFMove(error));
}

void RTCDataChannelRemoteHandler::bufferedAmountIsDecreasing(size_t amount)
{
    m_client->bufferedAmountIsDecreasing(amount);
}

void RTCDataChannelRemoteHandler::readyToSend()
{
    m_isReadyToSend = true;

    for (auto& message : m_pendingMessages)
        m_connection->sendData(m_remoteIdentifier, message.isRaw, message.buffer->makeContiguous()->span());
    m_pendingMessages.clear();

    if (m_isPendingClose)
        m_connection->close(m_remoteIdentifier);
}

void RTCDataChannelRemoteHandler::setClient(RTCDataChannelHandlerClient& client, std::optional<ScriptExecutionContextIdentifier> contextIdentifier)
{
    m_client = &client;
    m_connection->connectToSource(*this, contextIdentifier, *m_localIdentifier, m_remoteIdentifier);
}

bool RTCDataChannelRemoteHandler::sendStringData(const CString& text)
{
    if (!m_isReadyToSend) {
        m_pendingMessages.append(Message { false, SharedBuffer::create(text.span()) });
        return true;
    }
    m_connection->sendData(m_remoteIdentifier, false, text.span());
    return true;
}

bool RTCDataChannelRemoteHandler::sendRawData(std::span<const uint8_t> data)
{
    if (!m_isReadyToSend) {
        m_pendingMessages.append(Message { true, SharedBuffer::create(data) });
        return true;
    }
    m_connection->sendData(m_remoteIdentifier, true, data);
    return true;
}

void RTCDataChannelRemoteHandler::close()
{
    if (!m_isReadyToSend) {
        m_isPendingClose = true;
        return;
    }
    m_connection->close(m_remoteIdentifier);
}

} // namespace WebCore

#endif // ENABLE(WEB_RTC)
