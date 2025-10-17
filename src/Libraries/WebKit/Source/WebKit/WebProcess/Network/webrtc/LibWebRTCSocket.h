/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 25, 2025.
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

#if USE(LIBWEBRTC)

#include <WebCore/LibWebRTCProvider.h>
#include <WebCore/LibWebRTCSocketIdentifier.h>
#include <wtf/CheckedRef.h>
#include <wtf/Forward.h>
#include <wtf/Identified.h>
#include <wtf/StdMap.h>
#include <wtf/TZoneMalloc.h>

WTF_IGNORE_WARNINGS_IN_THIRD_PARTY_CODE_BEGIN
#include <webrtc/rtc_base/async_packet_socket.h>
WTF_IGNORE_WARNINGS_IN_THIRD_PARTY_CODE_END

namespace IPC {
class Connection;
class Decoder;
}

namespace WebKit {

class LibWebRTCSocketFactory;

class LibWebRTCSocket final : public rtc::AsyncPacketSocket, public CanMakeCheckedPtr<LibWebRTCSocket>, public Identified<WebCore::LibWebRTCSocketIdentifier> {
    WTF_MAKE_TZONE_ALLOCATED(LibWebRTCSocket);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(LibWebRTCSocket);
public:
    enum class Type { UDP, ClientTCP, ServerConnectionTCP };

    LibWebRTCSocket(LibWebRTCSocketFactory&, WebCore::ScriptExecutionContextIdentifier, Type, const rtc::SocketAddress& localAddress, const rtc::SocketAddress& remoteAddress);
    ~LibWebRTCSocket();

    WebCore::ScriptExecutionContextIdentifier contextIdentifier() const { return m_contextIdentifier; }
    const rtc::SocketAddress& localAddress() const { return m_localAddress; }
    const rtc::SocketAddress& remoteAddress() const { return m_remoteAddress; }

    void setError(int error) { m_error = error; }
    void setState(State state) { m_state = state; }

    void suspend();
    void resume();

private:
    bool willSend(size_t);

    friend class LibWebRTCNetwork;
    void signalReadPacket(std::span<const uint8_t>, rtc::SocketAddress&&, int64_t, rtc::EcnMarking);
    void signalSentPacket(int64_t, int64_t);
    void signalAddressReady(const rtc::SocketAddress&);
    void signalConnect();
    void signalClose(int);
    void signalUsedInterface(String&&);

    // AsyncPacketSocket API
    int GetError() const final { return m_error; }
    void SetError(int error) final { setError(error); }
    rtc::SocketAddress GetLocalAddress() const final;
    rtc::SocketAddress GetRemoteAddress() const final;
    int Send(const void *pv, size_t cb, const rtc::PacketOptions& options) final { return SendTo(pv, cb, m_remoteAddress, options); }
    int SendTo(const void *, size_t, const rtc::SocketAddress&, const rtc::PacketOptions&) final;
    int Close() final;
    State GetState() const final { return m_state; }
    int GetOption(rtc::Socket::Option, int*) final;
    int SetOption(rtc::Socket::Option, int) final;

    CheckedRef<LibWebRTCSocketFactory> m_factory;
    Type m_type;
    rtc::SocketAddress m_localAddress;
    rtc::SocketAddress m_remoteAddress;

    int m_error { 0 };
    State m_state { STATE_BINDING };

    StdMap<rtc::Socket::Option, int> m_options;

    bool m_isSuspended { false };
    WebCore::ScriptExecutionContextIdentifier m_contextIdentifier;
};

} // namespace WebKit

#endif // USE(LIBWEBRTC)
