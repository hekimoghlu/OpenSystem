/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 28, 2023.
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

#include "WebSocketIdentifier.h"
#include <wtf/Forward.h>
#include <wtf/Identified.h>
#include <wtf/Noncopyable.h>
#include <wtf/ObjectIdentifier.h>
#include <wtf/URL.h>

namespace JSC {
class ArrayBuffer;
}

namespace WebCore {

class Blob;
class Document;
class ResourceRequest;
class ResourceResponse;
class ScriptExecutionContext;
class SocketProvider;
class WebSocketChannel;
class WebSocketChannelInspector;
class WebSocketChannelClient;

using WebSocketChannelIdentifier = AtomicObjectIdentifier<WebSocketChannel>;

class ThreadableWebSocketChannel : public Identified<WebSocketIdentifier> {
    WTF_MAKE_NONCOPYABLE(ThreadableWebSocketChannel);
public:
    static RefPtr<ThreadableWebSocketChannel> create(Document&, WebSocketChannelClient&, SocketProvider&);
    static RefPtr<ThreadableWebSocketChannel> create(ScriptExecutionContext&, WebSocketChannelClient&, SocketProvider&);
    WEBCORE_EXPORT ThreadableWebSocketChannel();

    void ref() { refThreadableWebSocketChannel(); }
    void deref() { derefThreadableWebSocketChannel(); }

    enum class ConnectStatus { KO, OK };
    virtual ConnectStatus connect(const URL&, const String& protocol) = 0;
    virtual String subprotocol() = 0; // Will be available after didConnect() callback is invoked.
    virtual String extensions() = 0; // Will be available after didConnect() callback is invoked.

    virtual void send(CString&&) = 0;
    virtual void send(const JSC::ArrayBuffer&, unsigned byteOffset, unsigned byteLength) = 0;
    virtual void send(Blob&) = 0;

    virtual void close(int code, const String& reason) = 0;
    // Log the reason text and close the connection. Will call didClose().
    virtual void fail(String&& reason) = 0;
    virtual void disconnect() = 0; // Will suppress didClose().

    virtual void suspend() = 0;
    virtual void resume() = 0;

    virtual const WebSocketChannelInspector* channelInspector() const { return nullptr; };
    virtual WebSocketChannelIdentifier progressIdentifier() const = 0;
    virtual bool hasCreatedHandshake() const = 0;
    virtual bool isConnected() const = 0;
    using CookieGetter = Function<String(const URL&)>;
    virtual ResourceRequest clientHandshakeRequest(const CookieGetter&) const = 0;
    virtual const ResourceResponse& serverHandshakeResponse() const = 0;

    enum CloseEventCode {
        CloseEventCodeNotSpecified = -1,
        CloseEventCodeNormalClosure = 1000,
        CloseEventCodeGoingAway = 1001,
        CloseEventCodeProtocolError = 1002,
        CloseEventCodeUnsupportedData = 1003,
        CloseEventCodeFrameTooLarge = 1004,
        CloseEventCodeNoStatusRcvd = 1005,
        CloseEventCodeAbnormalClosure = 1006,
        CloseEventCodeInvalidFramePayloadData = 1007,
        CloseEventCodePolicyViolation = 1008,
        CloseEventCodeMessageTooBig = 1009,
        CloseEventCodeMandatoryExt = 1010,
        CloseEventCodeInternalError = 1011,
        CloseEventCodeTLSHandshake = 1015,
        CloseEventCodeMinimumUserDefined = 3000,
        CloseEventCodeMaximumUserDefined = 4999
    };
    
protected:
    virtual ~ThreadableWebSocketChannel() = default;
    virtual void refThreadableWebSocketChannel() = 0;
    virtual void derefThreadableWebSocketChannel() = 0;

    struct ValidatedURL {
        URL url;
        bool areCookiesAllowed { true };
    };
    WEBCORE_EXPORT static std::optional<ValidatedURL> validateURL(Document&, const URL&);
    WEBCORE_EXPORT static std::optional<ResourceRequest> webSocketConnectRequest(Document&, const URL&);
};

} // namespace WebCore
