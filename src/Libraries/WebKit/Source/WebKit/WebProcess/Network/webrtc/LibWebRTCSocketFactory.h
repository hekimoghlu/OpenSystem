/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 28, 2022.
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

#include "LibWebRTCResolver.h"
#include "LibWebRTCSocket.h"
#include "WebPageProxyIdentifier.h"
#include <WebCore/LibWebRTCMacros.h>
#include <WebCore/LibWebRTCSocketIdentifier.h>
#include <wtf/CheckedPtr.h>
#include <wtf/Deque.h>
#include <wtf/Function.h>
#include <wtf/HashMap.h>
#include <wtf/WeakRef.h>

WTF_IGNORE_WARNINGS_IN_THIRD_PARTY_CODE_BEGIN
#include <webrtc/rtc_base/net_helpers.h>
#include <webrtc/api/packet_socket_factory.h>
WTF_IGNORE_WARNINGS_IN_THIRD_PARTY_CODE_END

namespace WebKit {

class LibWebRTCNetwork;
class LibWebRTCSocket;

class LibWebRTCSocketFactory : public CanMakeThreadSafeCheckedPtr<LibWebRTCSocketFactory> {
    WTF_MAKE_FAST_ALLOCATED;
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(LibWebRTCSocketFactory);
public:
    LibWebRTCSocketFactory() = default;

    void addSocket(LibWebRTCSocket&);
    void removeSocket(LibWebRTCSocket&);
    LibWebRTCSocket* socket(WebCore::LibWebRTCSocketIdentifier identifier) { return m_sockets.get(identifier); }

    void forSocketInGroup(WebCore::ScriptExecutionContextIdentifier, const Function<void(LibWebRTCSocket&)>&);
    rtc::AsyncPacketSocket* createUdpSocket(WebCore::ScriptExecutionContextIdentifier, const rtc::SocketAddress&, uint16_t minPort, uint16_t maxPort, WebPageProxyIdentifier, bool isFirstParty, bool isRelayDisabled, const WebCore::RegistrableDomain&);
    rtc::AsyncPacketSocket* createClientTcpSocket(WebCore::ScriptExecutionContextIdentifier, const rtc::SocketAddress& localAddress, const rtc::SocketAddress& remoteAddress, String&& userAgent, const rtc::PacketSocketTcpOptions&, WebPageProxyIdentifier, bool isFirstParty, bool isRelayDisabled, const WebCore::RegistrableDomain&);

    CheckedPtr<LibWebRTCResolver> resolver(LibWebRTCResolverIdentifier identifier) { return m_resolvers.get(identifier); }
    void removeResolver(LibWebRTCResolverIdentifier identifier) { m_resolvers.remove(identifier); }

    std::unique_ptr<LibWebRTCResolver> createAsyncDnsResolver();

    void disableNonLocalhostConnections() { m_disableNonLocalhostConnections = true; }

    void setConnection(RefPtr<IPC::Connection>&&);
    IPC::Connection* connection();

private:
    // We cannot own sockets, clients of the factory are responsible to free them.
    HashMap<WebCore::LibWebRTCSocketIdentifier, CheckedRef<LibWebRTCSocket>> m_sockets;

    HashMap<LibWebRTCResolverIdentifier, CheckedRef<LibWebRTCResolver>> m_resolvers;
    bool m_disableNonLocalhostConnections { false };

    RefPtr<IPC::Connection> m_connection;
    Deque<Function<void(IPC::Connection&)>> m_pendingMessageTasks;
};

} // namespace WebKit

#endif // USE(LIBWEBRTC)
