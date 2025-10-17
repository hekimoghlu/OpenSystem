/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 21, 2025.
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

#include "Connection.h"
#include "LibWebRTCProvider.h"
#include "LibWebRTCSocketFactory.h"
#include "WebMDNSRegister.h"
#include "WebRTCMonitor.h"
#include "WebRTCResolver.h"
#include <WebCore/LibWebRTCSocketIdentifier.h>
#include <wtf/CheckedRef.h>
#include <wtf/FunctionDispatcher.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/UniqueRef.h>

namespace WebKit {
class WebProcess;

class LibWebRTCNetwork final : private FunctionDispatcher, public IPC::MessageReceiver {
    WTF_MAKE_TZONE_ALLOCATED(LibWebRTCNetwork);
public:
    explicit LibWebRTCNetwork(WebProcess&);
    ~LibWebRTCNetwork();

    void ref() const final;
    void deref() const final;

    IPC::Connection* connection() { return m_connection.get(); }
    void setConnection(RefPtr<IPC::Connection>&&);

    void networkProcessCrashed();

    bool isActive() const { return m_isActive; }

#if USE(LIBWEBRTC)
    WebRTCMonitor& monitor() { return m_webNetworkMonitor; }
    Ref<WebRTCMonitor> protectedMonitor() { return m_webNetworkMonitor; }
    LibWebRTCSocketFactory& socketFactory() { return m_socketFactory; }

    void disableNonLocalhostConnections() { socketFactory().disableNonLocalhostConnections(); }

    Ref<WebRTCResolver> resolver(LibWebRTCResolverIdentifier identifier) { return WebRTCResolver::create(socketFactory(), identifier); }
#endif

#if ENABLE(WEB_RTC)
    WebMDNSRegister& mdnsRegister() { return m_mdnsRegister; }
    Ref<WebMDNSRegister> protectedMDNSRegister() { return m_mdnsRegister; }
#endif

    void setAsActive();

private:
#if USE(LIBWEBRTC)
    void setSocketFactoryConnection();

    void signalReadPacket(WebCore::LibWebRTCSocketIdentifier, std::span<const uint8_t>, const RTCNetwork::IPAddress&, uint16_t port, int64_t, RTC::Network::EcnMarking);
    void signalSentPacket(WebCore::LibWebRTCSocketIdentifier, int64_t, int64_t);
    void signalAddressReady(WebCore::LibWebRTCSocketIdentifier, const RTCNetwork::SocketAddress&);
    void signalConnect(WebCore::LibWebRTCSocketIdentifier);
    void signalClose(WebCore::LibWebRTCSocketIdentifier, int);
    void signalUsedInterface(WebCore::LibWebRTCSocketIdentifier, String&&);
#endif

    // FunctionDispatcher
    void dispatch(Function<void()>&&) final;

#if USE(LIBWEBRTC)
    // IPC::MessageReceiver
    void didReceiveMessage(IPC::Connection&, IPC::Decoder&) final;
#endif

    CheckedRef<WebProcess> m_webProcess;

#if USE(LIBWEBRTC)
    LibWebRTCSocketFactory m_socketFactory;
    WebRTCMonitor m_webNetworkMonitor;
#endif
#if ENABLE(WEB_RTC)
    WebMDNSRegister m_mdnsRegister;
#endif
    bool m_isActive { false };
    RefPtr<IPC::Connection> m_connection;
};

} // namespace WebKit
