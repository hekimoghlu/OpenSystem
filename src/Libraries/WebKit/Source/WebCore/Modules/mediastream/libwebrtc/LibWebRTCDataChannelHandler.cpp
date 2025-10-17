/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 31, 2024.
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
#include "LibWebRTCDataChannelHandler.h"

#if ENABLE(WEB_RTC) && USE(LIBWEBRTC)

#include "EventNames.h"
#include "LibWebRTCUtils.h"
#include "RTCDataChannel.h"
#include "RTCError.h"
#include <wtf/MainThread.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(LibWebRTCDataChannelHandler);

template<typename T>
inline std::span<const T> span(const webrtc::DataBuffer& buffer)
{
    return unsafeMakeSpan(buffer.data.data<T>(), buffer.size());
}

webrtc::DataChannelInit LibWebRTCDataChannelHandler::fromRTCDataChannelInit(const RTCDataChannelInit& options)
{
    webrtc::DataChannelInit init;
    if (options.ordered)
        init.ordered = *options.ordered;
    if (options.maxPacketLifeTime)
        init.maxRetransmitTime = *options.maxPacketLifeTime;
    if (options.maxRetransmits)
        init.maxRetransmits = *options.maxRetransmits;
    init.protocol = options.protocol.utf8().data();
    if (options.negotiated)
        init.negotiated = *options.negotiated;
    if (options.id)
        init.id = *options.id;
    init.priority = webrtc::PriorityValue(fromRTCPriorityType(options.priority));
    return init;
}

LibWebRTCDataChannelHandler::LibWebRTCDataChannelHandler(rtc::scoped_refptr<webrtc::DataChannelInterface>&& channel)
    : m_channel(WTFMove(channel))
{
    ASSERT(m_channel);
    checkState();
    m_channel->RegisterObserver(this);
}

LibWebRTCDataChannelHandler::~LibWebRTCDataChannelHandler()
{
    m_channel->UnregisterObserver();
}

RTCDataChannelInit LibWebRTCDataChannelHandler::dataChannelInit() const
{
    auto protocol = m_channel->protocol();
    auto label = m_channel->label();

    RTCDataChannelInit init;
    init.ordered = m_channel->ordered();
    if (auto maxPacketLifeTime = m_channel->maxPacketLifeTime())
        init.maxPacketLifeTime = *maxPacketLifeTime;
    if (auto maxRetransmitsOpt = m_channel->maxRetransmitsOpt())
        init.maxRetransmits = *maxRetransmitsOpt;
    init.protocol = fromStdString(protocol);
    init.negotiated = m_channel->negotiated();
    init.id = m_channel->id();
    init.priority = toRTCPriorityType(m_channel->priority());
    return init;
}

String LibWebRTCDataChannelHandler::label() const
{
    return fromStdString(m_channel->label());
}

void LibWebRTCDataChannelHandler::setClient(RTCDataChannelHandlerClient& client, std::optional<ScriptExecutionContextIdentifier> contextIdentifier)
{
    Locker locker { m_clientLock };
    ASSERT(!m_client);
    ASSERT(!m_hasClient);
    m_hasClient = true;
    m_client = client;
    m_contextIdentifier = contextIdentifier;

    for (auto& message : m_bufferedMessages) {
        switchOn(message, [&](Ref<FragmentedSharedBuffer>& data) {
            Ref contiguousData = data->makeContiguous();
            client.didReceiveRawData(contiguousData->span());
        }, [&](String& text) {
            client.didReceiveStringData(text);
        }, [&](StateChange stateChange) {
            if (stateChange.error) {
                if (auto rtcError = toRTCError(*stateChange.error))
                    client.didDetectError(rtcError.releaseNonNull());
            }
            client.didChangeReadyState(stateChange.state);
        });
    }
    m_bufferedMessages.clear();
}

bool LibWebRTCDataChannelHandler::sendStringData(const CString& utf8Text)
{
    return m_channel->Send({ rtc::CopyOnWriteBuffer(utf8Text.data(), utf8Text.length()), false });
}

bool LibWebRTCDataChannelHandler::sendRawData(std::span<const uint8_t> data)
{
    return m_channel->Send({ rtc::CopyOnWriteBuffer(data.data(), data.size()), true });
}

void LibWebRTCDataChannelHandler::close()
{
    m_channel->Close();
}

std::optional<unsigned short> LibWebRTCDataChannelHandler::id() const
{
    auto id = m_channel->id();
    return id != -1 ? std::make_optional(id) : std::nullopt;
}

void LibWebRTCDataChannelHandler::OnStateChange()
{
    checkState();
}

void LibWebRTCDataChannelHandler::checkState()
{
    std::optional<webrtc::RTCError> error;
    RTCDataChannelState state;
    switch (m_channel->state()) {
    case webrtc::DataChannelInterface::kConnecting:
        state = RTCDataChannelState::Connecting;
        break;
    case webrtc::DataChannelInterface::kOpen:
        state = RTCDataChannelState::Open;
        break;
    case webrtc::DataChannelInterface::kClosing:
        state = RTCDataChannelState::Closing;
        break;
    case webrtc::DataChannelInterface::kClosed:
        error = m_channel->error();
        state = RTCDataChannelState::Closed;
        break;
    }

    Locker locker { m_clientLock };
    if (!m_hasClient) {
        m_bufferedMessages.append(StateChange { state, WTFMove(error) });
        return;
    }
    postTask([client = m_client, state, error = WTFMove(error)] {
        if (!client)
            return;
        if (error && !error->ok()) {
            auto rtcError = toRTCError(*error);
            if (!rtcError)
                rtcError = RTCError::create(RTCError::Init { RTCErrorDetailType::DataChannelFailure, { }, { }, { }, { } }, String { });
            client->didDetectError(rtcError.releaseNonNull());
        }
        client->didChangeReadyState(state);
    });
}

void LibWebRTCDataChannelHandler::OnMessage(const webrtc::DataBuffer& buffer)
{
    Locker locker { m_clientLock };
    if (!m_hasClient) {
        auto data = span<uint8_t>(buffer);
        if (buffer.binary)
            m_bufferedMessages.append(SharedBuffer::create(data));
        else
            m_bufferedMessages.append(String::fromUTF8(data));
        return;
    }

    std::unique_ptr<webrtc::DataBuffer> protectedBuffer(new webrtc::DataBuffer(buffer));
    postTask([client = m_client, buffer = WTFMove(protectedBuffer)] {
        if (!client)
            return;

        auto data = span<uint8_t>(*buffer);
        if (buffer->binary)
            client->didReceiveRawData(data);
        else
            client->didReceiveStringData(String::fromUTF8(data));
    });
}

void LibWebRTCDataChannelHandler::OnBufferedAmountChange(uint64_t amount)
{
    Locker locker { m_clientLock };
    if (!m_hasClient)
        return;

    postTask([client = m_client, amount] {
        if (client)
            client->bufferedAmountIsDecreasing(static_cast<size_t>(amount));
    });
}

void LibWebRTCDataChannelHandler::postTask(Function<void()>&& function)
{
    ASSERT(m_clientLock.isHeld());

    if (!m_contextIdentifier) {
        callOnMainThread(WTFMove(function));
        return;
    }
    ScriptExecutionContext::postTaskTo(*m_contextIdentifier, WTFMove(function));
}

} // namespace WebCore

#endif // ENABLE(WEB_RTC) && USE(LIBWEBRTC)
