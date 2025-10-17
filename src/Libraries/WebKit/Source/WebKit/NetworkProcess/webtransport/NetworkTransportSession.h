/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 25, 2025.
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

#include "MessageReceiver.h"
#include "MessageSender.h"
#include "WebPageProxyIdentifier.h"
#include <WebCore/ProcessQualified.h>
#include <wtf/Identified.h>
#include <wtf/RefCounted.h>
#include <wtf/TZoneMalloc.h>

#if PLATFORM(COCOA)
#include <Network/Network.h>
#include <wtf/RetainPtr.h>
#endif

namespace WebCore {
class Exception;
struct ClientOrigin;
struct WebTransportStreamIdentifierType;
using WebTransportStreamIdentifier = ObjectIdentifier<WebTransportStreamIdentifierType>;
using WebTransportSessionErrorCode = uint32_t;
using WebTransportStreamErrorCode = uint64_t;
}

namespace WebKit {

class NetworkConnectionToWebProcess;
class NetworkTransportStream;
enum class NetworkTransportStreamType : uint8_t;

struct SharedPreferencesForWebProcess;
struct WebTransportSessionIdentifierType;

using WebTransportSessionIdentifier = ObjectIdentifier<WebTransportSessionIdentifierType>;

class NetworkTransportSession : public RefCounted<NetworkTransportSession>, public IPC::MessageReceiver, public IPC::MessageSender, public Identified<WebTransportSessionIdentifier> {
    WTF_MAKE_TZONE_ALLOCATED(NetworkTransportSession);
public:
    static void initialize(NetworkConnectionToWebProcess&, URL&&, WebKit::WebPageProxyIdentifier&&, WebCore::ClientOrigin&&, CompletionHandler<void(RefPtr<NetworkTransportSession>&&)>&&);

    ~NetworkTransportSession();

    void ref() const final { RefCounted::ref(); }
    void deref() const final { RefCounted::deref(); }

    void sendDatagram(std::span<const uint8_t>, CompletionHandler<void(std::optional<WebCore::Exception>&&)>&&);
    void createOutgoingUnidirectionalStream(CompletionHandler<void(std::optional<WebCore::WebTransportStreamIdentifier>)>&&);
    void createBidirectionalStream(CompletionHandler<void(std::optional<WebCore::WebTransportStreamIdentifier>)>&&);
    void destroyOutgoingUnidirectionalStream(WebCore::WebTransportStreamIdentifier);
    void destroyBidirectionalStream(WebCore::WebTransportStreamIdentifier);
    void streamSendBytes(WebCore::WebTransportStreamIdentifier, std::span<const uint8_t>, bool withFin, CompletionHandler<void(std::optional<WebCore::Exception>&&)>&&);
    void terminate(WebCore::WebTransportSessionErrorCode, CString&&);

    void receiveDatagram(std::span<const uint8_t>, bool, std::optional<WebCore::Exception>&&);
    void streamReceiveBytes(WebCore::WebTransportStreamIdentifier, std::span<const uint8_t>, bool, std::optional<WebCore::Exception>&&);
    void receiveIncomingUnidirectionalStream(WebCore::WebTransportStreamIdentifier);
    void receiveBidirectionalStream(WebCore::WebTransportStreamIdentifier);

    void cancelReceiveStream(WebCore::WebTransportStreamIdentifier, std::optional<WebCore::WebTransportStreamErrorCode>);
    void cancelSendStream(WebCore::WebTransportStreamIdentifier, std::optional<WebCore::WebTransportStreamErrorCode>);
    void destroyStream(WebCore::WebTransportStreamIdentifier, std::optional<WebCore::WebTransportStreamErrorCode>);

    void didReceiveMessage(IPC::Connection&, IPC::Decoder&) final;
    std::optional<SharedPreferencesForWebProcess> sharedPreferencesForWebProcess() const;
private:
    template<typename... Args> static Ref<NetworkTransportSession> create(Args&&... args) { return adoptRef(*new NetworkTransportSession(std::forward<Args>(args)...)); }
#if PLATFORM(COCOA)
    NetworkTransportSession(NetworkConnectionToWebProcess&, nw_connection_group_t, nw_endpoint_t);
#endif

    IPC::Connection* messageSenderConnection() const final;
    uint64_t messageSenderDestinationID() const final;
    void setupConnectionHandler();
    void setupDatagramConnection(CompletionHandler<void(RefPtr<NetworkTransportSession>&&)>&&);
    void receiveDatagramLoop();
    void createStream(NetworkTransportStreamType, CompletionHandler<void(std::optional<WebCore::WebTransportStreamIdentifier>)>&&);

    HashMap<WebCore::WebTransportStreamIdentifier, Ref<NetworkTransportStream>> m_streams;
    WeakPtr<NetworkConnectionToWebProcess> m_connectionToWebProcess;

#if PLATFORM(COCOA)
    const RetainPtr<nw_connection_group_t> m_connectionGroup;
    const RetainPtr<nw_endpoint_t> m_endpoint;
    RetainPtr<nw_connection_t> m_datagramConnection;
#endif
};

}
