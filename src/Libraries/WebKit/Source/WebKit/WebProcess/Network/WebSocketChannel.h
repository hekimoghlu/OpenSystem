/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 28, 2022.
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
#include <WebCore/NetworkSendQueue.h>
#include <WebCore/ResourceRequest.h>
#include <WebCore/ResourceResponse.h>
#include <WebCore/ThreadableWebSocketChannel.h>
#include <WebCore/WebSocketChannelInspector.h>
#include <WebCore/WebSocketFrame.h>
#include <wtf/WeakPtr.h>

namespace IPC {
class Connection;
class Decoder;
}

namespace WebCore {
class WeakPtrImplWithEventTargetData;
}

namespace WebKit {

class WebSocketChannel : public IPC::MessageSender, public IPC::MessageReceiver, public WebCore::ThreadableWebSocketChannel, public RefCounted<WebSocketChannel> {
public:
    static Ref<WebSocketChannel> create(WebPageProxyIdentifier, WebCore::Document&, WebCore::WebSocketChannelClient&);
    ~WebSocketChannel();

    void ref() const final { RefCounted::ref(); }
    void deref() const final { RefCounted::deref(); }

    void didReceiveMessage(IPC::Connection&, IPC::Decoder&);

    void networkProcessCrashed();

    using RefCounted<WebSocketChannel>::ref;
    using RefCounted<WebSocketChannel>::deref;

private:
    WebSocketChannel(WebPageProxyIdentifier, WebCore::Document&, WebCore::WebSocketChannelClient&);

    static WebCore::NetworkSendQueue createMessageQueue(WebCore::Document&, WebSocketChannel&);

    // ThreadableWebSocketChannel
    ConnectStatus connect(const URL&, const String& protocol) final;
    String subprotocol() final;
    String extensions() final;
    void send(CString&&) final;
    void send(const JSC::ArrayBuffer&, unsigned byteOffset, unsigned byteLength) final;
    void send(WebCore::Blob&) final;
    void close(int code, const String& reason) final;
    void fail(String&& reason) final;
    void disconnect() final;
    void suspend() final;
    void resume() final;
    void refThreadableWebSocketChannel() final { ref(); }
    void derefThreadableWebSocketChannel() final { deref(); }

    void notifySendFrame(WebCore::WebSocketFrame::OpCode, std::span<const uint8_t> data);
    void logErrorMessage(const String&);

    // Message receivers
    void didConnect(String&& subprotocol, String&& extensions);
    void didReceiveText(String&&);
    void didReceiveBinaryData(std::span<const uint8_t>);
    void didClose(unsigned short code, String&&);
    void didReceiveMessageError(String&&);
    void didSendHandshakeRequest(WebCore::ResourceRequest&&);
    void didReceiveHandshakeResponse(WebCore::ResourceResponse&&);

    // MessageSender
    IPC::Connection* messageSenderConnection() const final;
    uint64_t messageSenderDestinationID() const final;

    bool increaseBufferedAmount(size_t);
    void decreaseBufferedAmount(size_t);
    template<typename T> void sendMessageInternal(T&&, size_t byteLength);

    const WebCore::WebSocketChannelInspector* channelInspector() const final { return &m_inspector; }
    WebCore::WebSocketChannelIdentifier progressIdentifier() const final { return m_inspector.progressIdentifier(); }
    bool hasCreatedHandshake() const final { return !m_url.isNull(); }
    bool isConnected() const final { return !m_handshakeResponse.isNull(); }
    WebCore::ResourceRequest clientHandshakeRequest(const CookieGetter&) const final { return m_handshakeRequest; }
    const WebCore::ResourceResponse& serverHandshakeResponse() const final { return m_handshakeResponse; }

    WeakPtr<WebCore::Document, WebCore::WeakPtrImplWithEventTargetData> m_document;
    ThreadSafeWeakPtr<WebCore::WebSocketChannelClient> m_client;
    URL m_url;
    String m_subprotocol;
    String m_extensions;
    size_t m_bufferedAmount { 0 };
    bool m_isClosing { false };
    WebCore::NetworkSendQueue m_messageQueue;
    WebCore::WebSocketChannelInspector m_inspector;
    WebCore::ResourceRequest m_handshakeRequest;
    WebCore::ResourceResponse m_handshakeResponse;
    WebPageProxyIdentifier m_webPageProxyID;
};

} // namespace WebKit
