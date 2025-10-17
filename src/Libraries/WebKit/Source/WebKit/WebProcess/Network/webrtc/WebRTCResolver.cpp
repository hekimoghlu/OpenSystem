/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 6, 2022.
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
#include "WebRTCResolver.h"

#if USE(LIBWEBRTC)

#include "LibWebRTCResolver.h"
#include "LibWebRTCSocketFactory.h"
#include <WebCore/LibWebRTCProvider.h>
#include <wtf/Function.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebKit {

WTF_MAKE_TZONE_ALLOCATED_IMPL(WebRTCResolver);

Ref<WebRTCResolver> WebRTCResolver::create(LibWebRTCSocketFactory& socketFactory, LibWebRTCResolverIdentifier identifier)
{
    return adoptRef(*new WebRTCResolver(socketFactory, identifier));
}

WebRTCResolver::WebRTCResolver(LibWebRTCSocketFactory& socketFactory, LibWebRTCResolverIdentifier identifier)
    : m_socketFactory(socketFactory)
    , m_identifier(identifier)
{
}

WebRTCResolver::~WebRTCResolver() = default;

void WebRTCResolver::setResolvedAddress(const Vector<RTCNetwork::IPAddress>& addresses)
{
    auto rtcAddresses = addresses.map([](auto& address) {
        return address.rtcAddress();
    });
    WebCore::LibWebRTCProvider::callOnWebRTCNetworkThread([factory = m_socketFactory, identifier = m_identifier, rtcAddresses = WTFMove(rtcAddresses)] () mutable {
        if (auto resolver = factory->resolver(identifier))
            resolver->setResolvedAddress(WTFMove(rtcAddresses));
    });
}

void WebRTCResolver::resolvedAddressError(int error)
{
    WebCore::LibWebRTCProvider::callOnWebRTCNetworkThread([factory = m_socketFactory, identifier = m_identifier, error]() {
        if (auto resolver = factory->resolver(identifier))
            resolver->setError(error);
    });
}

} // namespace WebKit

#endif // USE(LIBWEBRTC)
