/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 8, 2023.
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
#include <WebCore/FrameIdentifier.h>
#include <WebCore/PageIdentifier.h>
#include <WebCore/Timer.h>
#include <WebCore/WebSocketIdentifier.h>
#include <pal/SessionID.h>
#include <wtf/CompletionHandler.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/WeakPtr.h>

namespace WebCore {
struct ClientOrigin;
enum class AdvancedPrivacyProtections : uint16_t;
enum class StoredCredentialsPolicy : uint8_t;
class ResourceRequest;
class ResourceResponse;
}

namespace IPC {
class Connection;
class Decoder;
}

namespace WebKit {

class WebSocketTask;
class NetworkConnectionToWebProcess;
class NetworkProcess;
class NetworkSession;

class NetworkSocketChannel : public IPC::MessageSender, public IPC::MessageReceiver, public RefCounted<NetworkSocketChannel> {
    WTF_MAKE_TZONE_ALLOCATED(NetworkSocketChannel);
public:
    static RefPtr<NetworkSocketChannel> create(NetworkConnectionToWebProcess&, PAL::SessionID, const WebCore::ResourceRequest&, const String& protocol, WebCore::WebSocketIdentifier, WebPageProxyIdentifier, std::optional<WebCore::FrameIdentifier>, std::optional<WebCore::PageIdentifier>, const WebCore::ClientOrigin&, bool hadMainFrameMainResourcePrivateRelayed, bool allowPrivacyProxy, OptionSet<WebCore::AdvancedPrivacyProtections>, WebCore::StoredCredentialsPolicy);
    ~NetworkSocketChannel();

    void ref() const final { RefCounted::ref(); }
    void deref() const final { RefCounted::deref(); }

    void didReceiveMessage(IPC::Connection&, IPC::Decoder&);

    friend class WebSocketTask;

private:
    NetworkSocketChannel(NetworkConnectionToWebProcess&, NetworkSession*, const WebCore::ResourceRequest&, const String& protocol, WebCore::WebSocketIdentifier, WebPageProxyIdentifier, std::optional<WebCore::FrameIdentifier>, std::optional<WebCore::PageIdentifier>, const WebCore::ClientOrigin&, bool hadMainFrameMainResourcePrivateRelayed, bool allowPrivacyProxy, OptionSet<WebCore::AdvancedPrivacyProtections>, WebCore::StoredCredentialsPolicy);

    Ref<NetworkConnectionToWebProcess> protectedConnectionToWebProcess();

    void didConnect(const String& subprotocol, const String& extensions);
    void didReceiveText(const String&);
    void didReceiveBinaryData(std::span<const uint8_t>);
    void didClose(unsigned short code, const String& reason);
    void didReceiveMessageError(String&&);
    void didSendHandshakeRequest(WebCore::ResourceRequest&&);
    void didReceiveHandshakeResponse(WebCore::ResourceResponse&&);

    void sendString(std::span<const uint8_t>, CompletionHandler<void()>&&);
    void sendData(std::span<const uint8_t>, CompletionHandler<void()>&&);
    void close(int32_t code, const String& reason);
    void sendDelayedError();

    NetworkSession* session() const;

    CheckedPtr<WebSocketTask> checkedSocket();

    IPC::Connection* messageSenderConnection() const final;
    uint64_t messageSenderDestinationID() const final { return m_identifier.toUInt64(); }

    void finishClosingIfPossible();

    WeakRef<NetworkConnectionToWebProcess> m_connectionToWebProcess;
    WebCore::WebSocketIdentifier m_identifier;
    WeakPtr<NetworkSession> m_session;
    std::unique_ptr<WebSocketTask> m_socket;

    enum class State { Open, Closing, Closed };
    State m_state { State::Open };
    WebCore::Timer m_errorTimer;
    String m_errorMessage;
    std::optional<std::pair<unsigned short, String>> m_closeInfo;
    WebPageProxyIdentifier m_webPageProxyID;
};

} // namespace WebKit
