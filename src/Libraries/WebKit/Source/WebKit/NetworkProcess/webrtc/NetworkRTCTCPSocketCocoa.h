/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 6, 2025.
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

#if USE(LIBWEBRTC) && PLATFORM(COCOA)

#include "NetworkRTCProvider.h"
#include <Network/Network.h>
#include <wtf/TZoneMalloc.h>

namespace rtc {
class SocketAddress;
}

namespace WebKit {

class NetworkRTCTCPSocketCocoa final : public NetworkRTCProvider::Socket {
    WTF_MAKE_TZONE_ALLOCATED(NetworkRTCTCPSocketCocoa);
public:
    static std::unique_ptr<NetworkRTCProvider::Socket> createClientTCPSocket(WebCore::LibWebRTCSocketIdentifier, NetworkRTCProvider&, const rtc::SocketAddress&, int options, const String& attributedBundleIdentifier, bool isFirstParty, bool isRelayDisabled, const WebCore::RegistrableDomain&, Ref<IPC::Connection>&&);

    NetworkRTCTCPSocketCocoa(WebCore::LibWebRTCSocketIdentifier, NetworkRTCProvider&, const rtc::SocketAddress&, int options, const String& attributedBundleIdentifier, bool isFirstParty, bool isRelayDisabled, const WebCore::RegistrableDomain&, Ref<IPC::Connection>&&);
    ~NetworkRTCTCPSocketCocoa();

    static void getInterfaceName(NetworkRTCProvider&, const URL&, const String& attributedBundleIdentifier, bool isFirstParty, bool isRelayDisabled, const WebCore::RegistrableDomain&, CompletionHandler<void(String&&)>&&);

private:
    // NetworkRTCProvider::Socket.
    WebCore::LibWebRTCSocketIdentifier identifier() const final { return m_identifier; }
    Type type() const final { return Type::ClientTCP; }
    void close() final;
    void setOption(int option, int value) final;
    void sendTo(std::span<const uint8_t>, const rtc::SocketAddress&, const rtc::PacketOptions&) final;

    Vector<uint8_t> createMessageBuffer(std::span<const uint8_t>);

    WebCore::LibWebRTCSocketIdentifier m_identifier;
    CheckedRef<NetworkRTCProvider> m_rtcProvider;
    Ref<IPC::Connection> m_connection;
    RetainPtr<nw_connection_t> m_nwConnection;
    bool m_isSTUN { false };
#if ASSERT_ENABLED
    bool m_isClosed { false };
#endif
};

} // namespace WebKit

#endif // USE(LIBWEBRTC) && PLATFORM(COCOA)
