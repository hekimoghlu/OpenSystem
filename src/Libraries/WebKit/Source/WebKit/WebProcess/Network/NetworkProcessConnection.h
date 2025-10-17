/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 4, 2021.
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
#include <JavaScriptCore/ConsoleTypes.h>
#include <WebCore/FrameIdentifier.h>
#include <WebCore/MessagePortChannelProvider.h>
#include <WebCore/PageIdentifier.h>
#include <WebCore/RTCDataChannelIdentifier.h>
#include <WebCore/ResourceLoaderIdentifier.h>
#include <WebCore/ServiceWorkerTypes.h>
#include <WebCore/ShareableResource.h>
#include <wtf/RefCounted.h>
#include <wtf/text/WTFString.h>

namespace WebCore {
class ResourceError;
class ResourceRequest;
class ResourceResponse;
struct ClientOrigin;
struct Cookie;
struct MessagePortIdentifier;
struct MessageWithMessagePorts;
class SecurityOriginData;
enum class HTTPCookieAcceptPolicy : uint8_t;
}

namespace WebKit {

class WebIDBConnectionToServer;
class WebSharedWorkerObjectConnection;
class WebSWClientConnection;
class WebSharedWorkerObjectConnection;

enum class WebsiteDataType : uint32_t;

class NetworkProcessConnection final : public RefCounted<NetworkProcessConnection>, IPC::Connection::Client {
    WTF_MAKE_FAST_ALLOCATED;
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(NetworkProcessConnection);
public:
    static Ref<NetworkProcessConnection> create(IPC::Connection::Identifier&& connectionIdentifier, WebCore::HTTPCookieAcceptPolicy httpCookieAcceptPolicy)
    {
        return adoptRef(*new NetworkProcessConnection(WTFMove(connectionIdentifier), httpCookieAcceptPolicy));
    }
    ~NetworkProcessConnection();

    void ref() const final { RefCounted::ref(); }
    void deref() const final { RefCounted::deref(); }
    
    Ref<IPC::Connection> protectedConnection() { return m_connection; }
    IPC::Connection& connection() { return m_connection.get(); }

    void writeBlobsToTemporaryFilesForIndexedDB(const Vector<String>& blobURLs, CompletionHandler<void(Vector<String>&& filePaths)>&&);

    WebIDBConnectionToServer* existingIDBConnectionToServer() const { return m_webIDBConnection.get(); };
    WebIDBConnectionToServer& idbConnectionToServer();

    WebSWClientConnection& serviceWorkerConnection();
    WebSharedWorkerObjectConnection& sharedWorkerConnection();

#if HAVE(AUDIT_TOKEN)
    void setNetworkProcessAuditToken(std::optional<audit_token_t> auditToken) { m_networkProcessAuditToken = auditToken; }
    std::optional<audit_token_t> networkProcessAuditToken() const { return m_networkProcessAuditToken; }
#endif

    WebCore::HTTPCookieAcceptPolicy cookieAcceptPolicy() const { return m_cookieAcceptPolicy; }
    bool cookiesEnabled() const;

#if HAVE(COOKIE_CHANGE_LISTENER_API)
    void cookiesAdded(const String& host, Vector<WebCore::Cookie>&&);
    void cookiesDeleted(const String& host, Vector<WebCore::Cookie>&&);
    void allCookiesDeleted();
#endif
    void updateCachedCookiesEnabled();
    void loadCancelledDownloadRedirectRequestInFrame(WebCore::ResourceRequest&&, WebCore::FrameIdentifier, WebCore::PageIdentifier);
private:
    NetworkProcessConnection(IPC::Connection::Identifier&&, WebCore::HTTPCookieAcceptPolicy);

    // IPC::Connection::Client
    void didReceiveMessage(IPC::Connection&, IPC::Decoder&) override;
    bool didReceiveSyncMessage(IPC::Connection&, IPC::Decoder&, UniqueRef<IPC::Encoder>&) override;
    void didClose(IPC::Connection&) override;
    void didReceiveInvalidMessage(IPC::Connection&, IPC::MessageName, int32_t) override;
    bool dispatchMessage(IPC::Connection&, IPC::Decoder&);
    bool dispatchSyncMessage(IPC::Connection&, IPC::Decoder&, UniqueRef<IPC::Encoder>&);

    void didFinishPingLoad(WebCore::ResourceLoaderIdentifier pingLoadIdentifier, WebCore::ResourceError&&, WebCore::ResourceResponse&&);
    void didFinishPreconnection(WebCore::ResourceLoaderIdentifier preconnectionIdentifier, WebCore::ResourceError&&);
    void setOnLineState(bool isOnLine);
    void cookieAcceptPolicyChanged(WebCore::HTTPCookieAcceptPolicy);

    void messagesAvailableForPort(const WebCore::MessagePortIdentifier&);

#if ENABLE(SHAREABLE_RESOURCE)
    // Message handlers.
    void didCacheResource(const WebCore::ResourceRequest&, WebCore::ShareableResource::Handle&&);
#endif
#if ENABLE(WEB_RTC)
    void connectToRTCDataChannelRemoteSource(WebCore::RTCDataChannelIdentifier source, WebCore::RTCDataChannelIdentifier handler, CompletionHandler<void(std::optional<bool>)>&&);
#endif

    void broadcastConsoleMessage(MessageSource, MessageLevel, const String& message);

    // The connection from the web process to the network process.
    Ref<IPC::Connection> m_connection;
#if HAVE(AUDIT_TOKEN)
    std::optional<audit_token_t> m_networkProcessAuditToken;
#endif

    RefPtr<WebIDBConnectionToServer> m_webIDBConnection;

    RefPtr<WebSWClientConnection> m_swConnection;
    RefPtr<WebSharedWorkerObjectConnection> m_sharedWorkerConnection;
    WebCore::HTTPCookieAcceptPolicy m_cookieAcceptPolicy;
};

} // namespace WebKit
