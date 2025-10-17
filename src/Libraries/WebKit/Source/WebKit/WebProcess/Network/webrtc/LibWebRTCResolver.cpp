/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 10, 2025.
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
#include "LibWebRTCResolver.h"

#if USE(LIBWEBRTC)

#include "LibWebRTCNetwork.h"
#include "Logging.h"
#include "NetworkProcessConnection.h"
#include "NetworkRTCProviderMessages.h"
#include "WebProcess.h"
#include <wtf/MainThread.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebKit {

WTF_MAKE_TZONE_ALLOCATED_IMPL(LibWebRTCResolver);

void LibWebRTCResolver::sendOnMainThread(Function<void(IPC::Connection&)>&& callback)
{
    callOnMainRunLoop([callback = WTFMove(callback)]() {
        Ref networkProcessConnection = WebProcess::singleton().ensureNetworkProcessConnection().connection();
        callback(networkProcessConnection);
    });
}

LibWebRTCResolver::~LibWebRTCResolver()
{
    WebProcess::singleton().libWebRTCNetwork().socketFactory().removeResolver(identifier());
    sendOnMainThread([identifier = this->identifier()](IPC::Connection& connection) {
        connection.send(Messages::NetworkRTCProvider::StopResolver(identifier), 0);
    });
}

void LibWebRTCResolver::start(const rtc::SocketAddress& address, Function<void()>&& callback)
{
    ASSERT(!m_callback);

    m_callback = WTFMove(callback);
    m_addressToResolve = address;
    m_port = address.port();

    auto addressString = address.HostAsURIString();
    String name { std::span { addressString } };

    if (name.endsWithIgnoringASCIICase(".local"_s) && !WTF::isVersion4UUID(StringView { name }.left(name.length() - 6))) {
        RELEASE_LOG_ERROR(WebRTC, "mDNS candidate is not a Version 4 UUID");
        setError(-1);
        return;
    }

    sendOnMainThread([identifier = this->identifier(), name = WTFMove(name).isolatedCopy()](IPC::Connection& connection) {
        connection.send(Messages::NetworkRTCProvider::CreateResolver(identifier, name), 0);
    });
}

const webrtc::AsyncDnsResolverResult& LibWebRTCResolver::result() const
{
    return *this;
}

bool LibWebRTCResolver::GetResolvedAddress(int family, rtc::SocketAddress* address) const
{
    ASSERT(address);
    if (m_error || !m_addresses.size())
        return false;

    *address = m_addressToResolve;
    for (auto& ipAddress : m_addresses) {
        if (family == ipAddress.family()) {
            address->SetResolvedIP(ipAddress);
            address->SetPort(m_port);
            return true;
        }
    }
    return false;
}

void LibWebRTCResolver::setResolvedAddress(Vector<rtc::IPAddress>&& addresses)
{
    m_addresses = WTFMove(addresses);
    m_callback();
}

void LibWebRTCResolver::setError(int error)
{
    m_error = error;
    m_callback();
}

} // namespace WebKit

#endif // USE(LIBWEBRTC)
