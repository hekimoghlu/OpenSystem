/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 3, 2024.
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
#pragma once

#if !PLATFORM(COCOA)

#if USE(LIBWEBRTC)

#include "NetworkRTCProvider.h"

WTF_IGNORE_WARNINGS_IN_THIRD_PARTY_CODE_BEGIN
#include <webrtc/rtc_base/async_packet_socket.h>
WTF_IGNORE_WARNINGS_IN_THIRD_PARTY_CODE_END

#include <wtf/TZoneMalloc.h>

namespace rtc {
class AsyncPacketSocket;
struct SentPacket;
typedef int64_t PacketTime;
}

namespace WebKit {

class LibWebRTCSocketClient final : public NetworkRTCProvider::Socket, public sigslot::has_slots<> {
    WTF_MAKE_TZONE_ALLOCATED(LibWebRTCSocketClient);
public:
    LibWebRTCSocketClient(WebCore::LibWebRTCSocketIdentifier, NetworkRTCProvider&, std::unique_ptr<rtc::AsyncPacketSocket>&&, Type, Ref<IPC::Connection>&&);

private:
    WebCore::LibWebRTCSocketIdentifier identifier() const final { return m_identifier; }
    Type type() const final { return m_type; }
    void close() final;

    void setOption(int option, int value) final;
    void sendTo(std::span<const uint8_t>, const rtc::SocketAddress&, const rtc::PacketOptions&) final;

    void signalReadPacket(rtc::AsyncPacketSocket*, const unsigned char*, size_t, const rtc::SocketAddress&, int64_t);
    void signalSentPacket(rtc::AsyncPacketSocket*, const rtc::SentPacket&);
    void signalAddressReady(rtc::AsyncPacketSocket*, const rtc::SocketAddress&);
    void signalConnect(rtc::AsyncPacketSocket*);
    void signalClose(rtc::AsyncPacketSocket*, int);

    void signalAddressReady();

    WebCore::LibWebRTCSocketIdentifier m_identifier;
    Type m_type;
    CheckedRef<NetworkRTCProvider> m_rtcProvider;
    std::unique_ptr<rtc::AsyncPacketSocket> m_socket;
    Ref<IPC::Connection> m_connection;
    int m_sendError { 0 };
};

} // namespace WebKit

#endif // USE(LIBWEBRTC)

#endif // !PLATFORM(COCOA)
