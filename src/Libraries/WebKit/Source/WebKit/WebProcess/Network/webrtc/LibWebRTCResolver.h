/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 5, 2025.
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

#include "LibWebRTCDnsResolverFactory.h"
#include "LibWebRTCResolverIdentifier.h"
#include <WebCore/LibWebRTCMacros.h>
#include <wtf/CheckedPtr.h>
#include <wtf/Identified.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/Vector.h>

namespace IPC {
class Connection;
}

namespace WebKit {
class LibWebRTCSocketFactory;

class LibWebRTCResolver final : public LibWebRTCDnsResolverFactory::Resolver, private webrtc::AsyncDnsResolverResult, public CanMakeCheckedPtr<LibWebRTCResolver>, public Identified<LibWebRTCResolverIdentifier> {
    WTF_MAKE_TZONE_ALLOCATED(LibWebRTCResolver);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(LibWebRTCResolver);
public:
    LibWebRTCResolver() = default;
    ~LibWebRTCResolver();

    void start(const rtc::SocketAddress&, Function<void()>&&) final;

private:
    friend class WebRTCResolver;

    // webrtc::AsyncDnsResolverInterface
    const webrtc::AsyncDnsResolverResult& result() const final;

    // webrtc::AsyncDnsResolverResult
    bool GetResolvedAddress(int family, rtc::SocketAddress*) const final;
    int GetError() const { return m_error; }

    void setError(int);
    void setResolvedAddress(Vector<rtc::IPAddress>&&);

    static void sendOnMainThread(Function<void(IPC::Connection&)>&&);

    Vector<rtc::IPAddress> m_addresses;
    rtc::SocketAddress m_addressToResolve;
    Function<void()> m_callback;
    int m_error { 0 };
    uint16_t m_port { 0 };
};

} // namespace WebKit

#endif // USE(LIBWEBRTC)
