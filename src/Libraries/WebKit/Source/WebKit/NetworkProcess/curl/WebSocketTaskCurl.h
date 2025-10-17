/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 28, 2024.
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

#include "NetworkSessionCurl.h"
#include "WebSocketTask.h"
#include <WebCore/CurlStream.h>
#include <WebCore/ResourceRequest.h>
#include <WebCore/ThreadableWebSocketChannel.h>
#include <WebCore/WebSocketDeflateFramer.h>
#include <WebCore/WebSocketFrame.h>
#include <wtf/TZoneMalloc.h>

namespace WebKit {
class WebSocketTask;
}

namespace WebCore {
class CertificateInfo;
class CurlStreamScheduler;
class SharedBuffer;
class WebSocketHandshake;
struct ClientOrigin;
}

namespace WebKit {

class NetworkSocketChannel;
struct SessionSet;

class WebSocketTask : public CanMakeWeakPtr<WebSocketTask>, public CanMakeCheckedPtr<WebSocketTask>, public WebCore::CurlStream::Client {
    WTF_MAKE_TZONE_ALLOCATED(WebSocketTask);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(WebSocketTask);
public:
    WebSocketTask(NetworkSocketChannel&, WebPageProxyIdentifier, const WebCore::ResourceRequest&, const String& protocol, const WebCore::ClientOrigin&);
    virtual ~WebSocketTask();

    void sendString(std::span<const uint8_t>, CompletionHandler<void()>&&);
    void sendData(std::span<const uint8_t>, CompletionHandler<void()>&&);
    void close(int32_t code, const String& reason);

    void cancel();
    void resume();

    NetworkSessionCurl* networkSession();
    SessionSet* sessionSet() { return nullptr; }

    WebPageProxyIdentifier webPageProxyID() const { return m_webProxyPageID; }

    const WebCore::SecurityOriginData& topOrigin() const { return m_topOrigin; }

private:
    enum class State : uint8_t {
        Connecting,
        Handshaking,
        Opened,
        Closing,
        Closed
    };

    void didOpen(WebCore::CurlStreamID) final;
    void didSendData(WebCore::CurlStreamID, size_t) final { };
    void didReceiveData(WebCore::CurlStreamID, const WebCore::SharedBuffer&) final;
    void didFail(WebCore::CurlStreamID, CURLcode, WebCore::CertificateInfo&&) final;

    Ref<NetworkSocketChannel> protectedChannel() const;

    void tryServerTrustEvaluation(WebCore::AuthenticationChallenge&&, String&&);

    bool appendReceivedBuffer(const WebCore::SharedBuffer&);
    void skipReceivedBuffer(size_t len);

    Expected<bool, String> validateOpeningHandshake();
    std::optional<String> receiveFrames(Function<void(WebCore::WebSocketFrame::OpCode, std::span<const uint8_t>)>&&);
    std::optional<String> validateFrame(const WebCore::WebSocketFrame&);

    bool sendFrame(WebCore::WebSocketFrame::OpCode, std::span<const uint8_t> data);
    void sendClosingHandshakeIfNeeded(int32_t, const String& reason);

    void didFail(String&& reason);
    void didClose(int32_t code, const String& reason);

    bool isStreamInvalidated() { return m_streamID == WebCore::invalidCurlStreamID; }
    void destructStream();

    WeakRef<NetworkSocketChannel> m_channel;
    WebPageProxyIdentifier m_webProxyPageID;
    WebCore::ResourceRequest m_request;
    String m_protocol;
    WebCore::SecurityOriginData m_topOrigin;

    WebCore::CurlStreamScheduler& m_scheduler;
    WebCore::CurlStreamID m_streamID { WebCore::invalidCurlStreamID };

    State m_state { State::Connecting };

    std::unique_ptr<WebCore::WebSocketHandshake> m_handshake;
    WebCore::WebSocketDeflateFramer m_deflateFramer;

    bool m_didCompleteOpeningHandshake { false };

    bool m_shouldDiscardReceivedData { false };
    Vector<uint8_t> m_receiveBuffer;

    bool m_hasContinuousFrame { false };
    WebCore::WebSocketFrame::OpCode m_continuousFrameOpCode { WebCore::WebSocketFrame::OpCode::OpCodeInvalid };
    Vector<uint8_t> m_continuousFrameData;

    bool m_receivedClosingHandshake { false };
    int32_t m_closeEventCode { WebCore::ThreadableWebSocketChannel::CloseEventCode::CloseEventCodeNotSpecified };
    String m_closeEventReason;

    bool m_didSendClosingHandshake { false };
    bool m_receivedDidFail { false };
};

} // namespace WebKit
