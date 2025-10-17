/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 5, 2023.
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
#include <limits>
#include <wtf/HashMap.h>
#include <wtf/Lock.h>
#include <wtf/TZoneMalloc.h>

namespace rtc {
class SocketAddress;
}

namespace WTF {

template<> struct DefaultHash<rtc::SocketAddress> {
    static unsigned hash(const rtc::SocketAddress& address) { return address.Hash(); }
    static bool equal(const rtc::SocketAddress& a, const rtc::SocketAddress& b) { return a == b || (a.IsNil() && b.IsNil() && a.scope_id() == b.scope_id()); }
    static constexpr bool safeToCompareToEmptyOrDeleted = true;
};

template<> struct HashTraits<rtc::SocketAddress> : GenericHashTraits<rtc::SocketAddress> {
    static rtc::SocketAddress emptyValue() { return rtc::SocketAddress { }; }
    static void constructDeletedValue(rtc::SocketAddress& address) { address.SetScopeID(std::numeric_limits<int>::min()); }
    static bool isDeletedValue(const rtc::SocketAddress& address) { return address.scope_id() == std::numeric_limits<int>::min(); }
};

}

namespace WebKit {

class NetworkRTCUDPSocketCocoaConnections;

class NetworkRTCUDPSocketCocoa final : public NetworkRTCProvider::Socket {
    WTF_MAKE_TZONE_ALLOCATED(NetworkRTCUDPSocketCocoa);
public:
    static std::unique_ptr<NetworkRTCProvider::Socket> createUDPSocket(WebCore::LibWebRTCSocketIdentifier, NetworkRTCProvider&, const rtc::SocketAddress&, uint16_t minPort, uint16_t maxPort, Ref<IPC::Connection>&&, String&& attributedBundleIdentifier, bool isFirstParty, bool isRelayDisabled, const WebCore::RegistrableDomain&);

    NetworkRTCUDPSocketCocoa(WebCore::LibWebRTCSocketIdentifier, NetworkRTCProvider&, const rtc::SocketAddress&, Ref<IPC::Connection>&&, String&& attributedBundleIdentifier, bool isFirstParty, bool isRelayDisabled, const WebCore::RegistrableDomain&);
    ~NetworkRTCUDPSocketCocoa();

private:
    // NetworkRTCProvider::Socket.
    WebCore::LibWebRTCSocketIdentifier identifier() const final { return m_identifier; }
    Type type() const final { return Type::UDP; }
    void close() final;
    void setOption(int option, int value) final;
    void sendTo(std::span<const uint8_t>, const rtc::SocketAddress&, const rtc::PacketOptions&) final;

    CheckedRef<NetworkRTCProvider> m_rtcProvider;
    WebCore::LibWebRTCSocketIdentifier m_identifier;
    Ref<NetworkRTCUDPSocketCocoaConnections> m_connections;
};

} // namespace WebKit

#endif // USE(LIBWEBRTC) && PLATFORM(COCOA)
