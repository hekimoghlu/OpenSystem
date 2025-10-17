/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 30, 2025.
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
#include "RTCNetwork.h"

#include <wtf/CrossThreadCopier.h>
#include <wtf/StdLibExtras.h>

#if USE(LIBWEBRTC)

namespace WebKit {

RTCNetwork::RTCNetwork(Vector<char>&& name, Vector<char>&& description, IPAddress prefix, int prefixLength, int type, uint16_t id, int preference, bool active, bool ignored, int scopeID, Vector<InterfaceAddress>&& ips)
    : name(WTFMove(name))
    , description(WTFMove(description))
    , prefix(prefix)
    , prefixLength(prefixLength)
    , type(type)
    , id(id)
    , preference(preference)
    , active(active)
    , ignored(ignored)
    , scopeID(scopeID)
    , ips(WTFMove(ips)) { }

rtc::Network RTCNetwork::value() const
{
    rtc::Network network({ name.data(), name.size() }, { description.data(), description.size() }, prefix.rtcAddress(), prefixLength, rtc::AdapterType(type));
    network.set_id(id);
    network.set_preference(preference);
    network.set_active(active);
    network.set_ignored(ignored);
    network.set_scope_id(scopeID);

    std::vector<rtc::InterfaceAddress> vector;
    vector.reserve(ips.size());
    for (auto& ip : ips)
        vector.push_back(ip.rtcAddress());
    network.SetIPs(WTFMove(vector), true);

    return network;
}

RTCNetwork RTCNetwork::isolatedCopy() const
{
    return RTCNetwork {
        crossThreadCopy(name),
        crossThreadCopy(description),
        prefix,
        prefixLength,
        type,
        id,
        preference,
        active,
        ignored,
        scopeID,
        crossThreadCopy(ips)
    };
}

namespace RTC::Network {

rtc::SocketAddress SocketAddress::rtcAddress() const
{
    rtc::SocketAddress result;
    result.SetPort(port);
    result.SetScopeID(scopeID);
    result.SetIP({ hostname.data(), hostname.size() });
    if (ipAddress)
        result.SetResolvedIP(ipAddress->rtcAddress());
    return result;
}

SocketAddress::SocketAddress(const rtc::SocketAddress& value)
    : port(value.port())
    , scopeID(value.scope_id())
    , hostname(std::span { value.hostname() })
    , ipAddress(value.IsUnresolvedIP() ? std::nullopt : std::optional(IPAddress(value.ipaddr())))
{
}

static std::array<uint32_t, 4> fromIPv6Address(const struct in6_addr& address)
{
    std::array<uint32_t, 4> array;
    static_assert(sizeof(array) == sizeof(address));
    memcpySpan(asMutableByteSpan(array), asByteSpan(address));
    return array;
}

IPAddress::IPAddress(const rtc::IPAddress& input)
{
    switch (input.family()) {
    case AF_INET6:
        value = fromIPv6Address(input.ipv6_address());
        break;
    case AF_INET:
        value = input.ipv4_address().s_addr;
        break;
    case AF_UNSPEC:
        value = UnspecifiedFamily { };
        break;
    default:
        RELEASE_ASSERT_NOT_REACHED();
    }
}

IPAddress::IPAddress(const struct sockaddr& address)
{
    switch (address.sa_family) {
    case AF_INET6:
        value = fromIPv6Address(reinterpret_cast<const sockaddr_in6*>(&address)->sin6_addr);
        break;
    case AF_INET:
        value = reinterpret_cast<const sockaddr_in*>(&address)->sin_addr.s_addr;
        break;
    case AF_UNSPEC:
        value = UnspecifiedFamily { };
        break;
    default:
        RELEASE_ASSERT_NOT_REACHED();
    }
}

rtc::IPAddress IPAddress::rtcAddress() const
{
    return WTF::switchOn(value, [](UnspecifiedFamily) {
        return rtc::IPAddress();
    }, [] (uint32_t ipv4) {
        in_addr addressv4;
        addressv4.s_addr = ipv4;
        return rtc::IPAddress(addressv4);
    }, [] (std::array<uint32_t, 4> ipv6) {
        in6_addr result;
        static_assert(sizeof(ipv6) == sizeof(result));
        memcpySpan(asMutableByteSpan(result), asByteSpan(ipv6));
        return rtc::IPAddress(result);
    });
}

rtc::InterfaceAddress InterfaceAddress::rtcAddress() const
{
    return rtc::InterfaceAddress(address.rtcAddress(), ipv6Flags);
}

}

}

#endif // USE(LIBWEBRTC)
