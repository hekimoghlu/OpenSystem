/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 25, 2022.
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

#include <WebCore/LibWebRTCMacros.h>
#include <optional>
#include <wtf/Forward.h>
#include <wtf/Vector.h>

WTF_IGNORE_WARNINGS_IN_THIRD_PARTY_CODE_BEGIN
#include <webrtc/rtc_base/socket_address.h>
#include <webrtc/rtc_base/network.h>
WTF_IGNORE_WARNINGS_IN_THIRD_PARTY_CODE_END

namespace WebKit {

namespace RTC::Network {

// This enums corresponds to rtc::EcnMarking.
enum class EcnMarking : int {
    kNotEct = 0, // Not ECN-Capable Transport
    kEct1 = 1, // ECN-Capable Transport
    kEct0 = 2, // Not used by L4s (or webrtc.)
    kCe = 3, // Congestion experienced
};

struct IPAddress {
    struct UnspecifiedFamily { };

    IPAddress() = default;
    explicit IPAddress(const rtc::IPAddress&);
    explicit IPAddress(const struct sockaddr&);
    explicit IPAddress(std::variant<UnspecifiedFamily, uint32_t, std::array<uint32_t, 4>> value)
        : value(value)
    {
    }

    IPAddress isolatedCopy() const { return *this; }
    rtc::IPAddress rtcAddress() const;

    bool isUnspecified() const { return std::holds_alternative<UnspecifiedFamily>(value); }

    std::variant<UnspecifiedFamily, uint32_t, std::array<uint32_t, 4>> value;
};

struct InterfaceAddress {
    explicit InterfaceAddress(IPAddress address, int ipv6Flags)
        : address(address), ipv6Flags(ipv6Flags)
    {
    }

    rtc::InterfaceAddress rtcAddress() const;
    InterfaceAddress isolatedCopy() const { return *this; }

    IPAddress address;
    int ipv6Flags;
};

struct SocketAddress {
    explicit SocketAddress(const rtc::SocketAddress&);
    explicit SocketAddress(uint16_t port, int scopeID, Vector<char>&& hostname, std::optional<IPAddress> ipAddress)
        : port(port)
        , scopeID(scopeID)
        , hostname(WTFMove(hostname))
        , ipAddress(ipAddress) { }

    rtc::SocketAddress rtcAddress() const;

    uint16_t port;
    int scopeID;
    Vector<char> hostname;
    std::optional<IPAddress> ipAddress;
};

}

struct RTCNetwork {
    using SocketAddress = RTC::Network::SocketAddress;
    using IPAddress = RTC::Network::IPAddress;
    using InterfaceAddress = RTC::Network::InterfaceAddress;

    RTCNetwork() = default;
    explicit RTCNetwork(Vector<char>&& name, Vector<char>&& description, IPAddress prefix, int prefixLength, int type, uint16_t id, int preference, bool active, bool ignored, int scopeID, Vector<InterfaceAddress>&& ips);
    RTCNetwork isolatedCopy() const;

    rtc::Network value() const;

    Vector<char> name;
    Vector<char> description;
    IPAddress prefix;
    int prefixLength { 0 };
    int type { 0 };
    uint16_t id { 0 };
    int preference { 0 };
    bool active { 0 };
    bool ignored { false };
    int scopeID { 0 };
    Vector<InterfaceAddress> ips;
};

}

#endif // USE(LIBWEBRTC)
