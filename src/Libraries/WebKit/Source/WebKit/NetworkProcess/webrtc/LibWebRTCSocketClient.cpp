/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 22, 2023.
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
#include "LibWebRTCSocketClient.h"

#if !PLATFORM(COCOA)

#if USE(LIBWEBRTC)

#include "Connection.h"
#include "LibWebRTCNetworkMessages.h"
#include "Logging.h"
#include "NetworkRTCProvider.h"
#include <WebCore/SharedBuffer.h>
#include <wtf/Function.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebKit {

WTF_MAKE_TZONE_ALLOCATED_IMPL(LibWebRTCSocketClient);

LibWebRTCSocketClient::LibWebRTCSocketClient(WebCore::LibWebRTCSocketIdentifier identifier, NetworkRTCProvider& rtcProvider, std::unique_ptr<rtc::AsyncPacketSocket>&& socket, Type type, Ref<IPC::Connection>&& connection)
    : m_identifier(identifier)
    , m_type(type)
    , m_rtcProvider(rtcProvider)
    , m_socket(WTFMove(socket))
    , m_connection(WTFMove(connection))
{
    ASSERT(m_socket);

    m_socket->RegisterReceivedPacketCallback([this](auto* socket, auto& packet) {
        signalReadPacket(socket, packet.payload().data(), packet.payload().size(), packet.source_address(), packet.arrival_time()->us_or(0));
    });
    m_socket->SignalSentPacket.connect(this, &LibWebRTCSocketClient::signalSentPacket);
    m_socket->SubscribeCloseEvent(this, [this](rtc::AsyncPacketSocket* socket, int error) {
        signalClose(socket, error);
    });

    switch (type) {
    case Type::ServerConnectionTCP:
        return;
    case Type::ClientTCP:
        m_socket->SignalConnect.connect(this, &LibWebRTCSocketClient::signalConnect);
        m_socket->SignalAddressReady.connect(this, &LibWebRTCSocketClient::signalAddressReady);
        return;
    case Type::UDP:
        m_socket->SignalConnect.connect(this, &LibWebRTCSocketClient::signalConnect);
        signalAddressReady();
        return;
    }
}

void LibWebRTCSocketClient::sendTo(std::span<const uint8_t> data, const rtc::SocketAddress& socketAddress, const rtc::PacketOptions& options)
{
    m_socket->SendTo(data.data(), data.size(), socketAddress, options);
    auto error = m_socket->GetError();
    RELEASE_LOG_ERROR_IF(error && m_sendError != error, Network, "LibWebRTCSocketClient::sendTo (ID=%" PRIu64 ") failed with error %d", m_identifier.toUInt64(), error);
    m_sendError = error;
}

void LibWebRTCSocketClient::close()
{
    ASSERT(m_socket);
    auto result = m_socket->Close();
    UNUSED_PARAM(result);
    RELEASE_LOG_ERROR_IF(result, Network, "LibWebRTCSocketClient::close (ID=%" PRIu64 ") failed with error %d", m_identifier.toUInt64(), m_socket->GetError());

    m_socket->DeregisterReceivedPacketCallback();
    Ref { m_rtcProvider.get() }->takeSocket(m_identifier);
}

void LibWebRTCSocketClient::setOption(int option, int value)
{
    ASSERT(m_socket);
    auto result = m_socket->SetOption(static_cast<rtc::Socket::Option>(option), value);
    UNUSED_PARAM(result);
    RELEASE_LOG_ERROR_IF(result, Network, "LibWebRTCSocketClient::setOption(%d, %d) (ID=%" PRIu64 ") failed with error %d", option, value, m_identifier.toUInt64(), m_socket->GetError());
}

void LibWebRTCSocketClient::signalReadPacket(rtc::AsyncPacketSocket* socket, const unsigned char* value, size_t length, const rtc::SocketAddress& address, int64_t packetTime)
{
    ASSERT_UNUSED(socket, m_socket.get() == socket);
    std::span data(byteCast<uint8_t>(value), length);
    m_connection->send(Messages::LibWebRTCNetwork::SignalReadPacket(m_identifier, data, RTCNetwork::IPAddress(address.ipaddr()), address.port(), packetTime, RTC::Network::EcnMarking::kNotEct), 0);
}

void LibWebRTCSocketClient::signalSentPacket(rtc::AsyncPacketSocket* socket, const rtc::SentPacket& sentPacket)
{
    ASSERT_UNUSED(socket, m_socket.get() == socket);
    m_connection->send(Messages::LibWebRTCNetwork::SignalSentPacket(m_identifier, sentPacket.packet_id, sentPacket.send_time_ms), 0);
}

void LibWebRTCSocketClient::signalAddressReady(rtc::AsyncPacketSocket* socket, const rtc::SocketAddress& address)
{
    ASSERT_UNUSED(socket, m_socket.get() == socket);
    m_connection->send(Messages::LibWebRTCNetwork::SignalAddressReady(m_identifier, RTCNetwork::SocketAddress(address)), 0);
}

void LibWebRTCSocketClient::signalAddressReady()
{
    signalAddressReady(m_socket.get(), m_socket->GetLocalAddress());
}

void LibWebRTCSocketClient::signalConnect(rtc::AsyncPacketSocket* socket)
{
    ASSERT_UNUSED(socket, m_socket.get() == socket);
    m_connection->send(Messages::LibWebRTCNetwork::SignalConnect(m_identifier), 0);
}

void LibWebRTCSocketClient::signalClose(rtc::AsyncPacketSocket* socket, int error)
{
    ASSERT_UNUSED(socket, m_socket.get() == socket);
    m_connection->send(Messages::LibWebRTCNetwork::SignalClose(m_identifier, error), 0);

    // We want to remove 'this' from the socket map now but we will destroy it asynchronously
    // so that the socket parameter of signalClose remains alive as the caller of signalClose may actually being using it afterwards.
    Ref rtcProvider = m_rtcProvider.get();
    rtcProvider->callOnRTCNetworkThread([socket = rtcProvider->takeSocket(m_identifier)] { });
}

} // namespace WebKit

#endif // USE(LIBWEBRTC)

#endif // !PLATFORM(COCOA)
