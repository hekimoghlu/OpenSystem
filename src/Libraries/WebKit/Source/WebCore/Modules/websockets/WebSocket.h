/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 12, 2025.
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

#include "ActiveDOMObject.h"
#include "EventTarget.h"
#include "ExceptionOr.h"
#include <wtf/URL.h>
#include "WebSocketChannelClient.h"
#include <wtf/HashSet.h>
#include <wtf/Lock.h>

namespace JSC {
class ArrayBuffer;
class ArrayBufferView;
}

namespace WebCore {

class Blob;
class ThreadableWebSocketChannel;

class WebSocket final : public RefCounted<WebSocket>, public EventTarget, public ActiveDOMObject, private WebSocketChannelClient {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(WebSocket);
public:
    void ref() const final { RefCounted::ref(); }
    void deref() const final { RefCounted::deref(); }

    static ASCIILiteral subprotocolSeparator();

    static ExceptionOr<Ref<WebSocket>> create(ScriptExecutionContext&, const String& url);
    static ExceptionOr<Ref<WebSocket>> create(ScriptExecutionContext&, const String& url, const String& protocol);
    static ExceptionOr<Ref<WebSocket>> create(ScriptExecutionContext&, const String& url, const Vector<String>& protocols);
    virtual ~WebSocket();

    static HashSet<WebSocket*>& allActiveWebSockets() WTF_REQUIRES_LOCK(s_allActiveWebSocketsLock);
    static Lock& allActiveWebSocketsLock() WTF_RETURNS_LOCK(s_allActiveWebSocketsLock);

    enum State {
        CONNECTING = 0,
        OPEN = 1,
        CLOSING = 2,
        CLOSED = 3
    };

    ExceptionOr<void> connect(const String& url);
    ExceptionOr<void> connect(const String& url, const String& protocol);
    ExceptionOr<void> connect(const String& url, const Vector<String>& protocols);

    ExceptionOr<void> send(const String& message);
    ExceptionOr<void> send(JSC::ArrayBuffer&);
    ExceptionOr<void> send(JSC::ArrayBufferView&);
    ExceptionOr<void> send(Blob&);

    ExceptionOr<void> close(std::optional<unsigned short> code, const String& reason);

    RefPtr<ThreadableWebSocketChannel> channel() const;

    const URL& url() const;
    State readyState() const;
    unsigned bufferedAmount() const;

    String protocol() const;
    String extensions() const;

    enum class BinaryType : bool { Blob, Arraybuffer };
    BinaryType binaryType() const { return m_binaryType; }
    void setBinaryType(BinaryType);

    ScriptExecutionContext* scriptExecutionContext() const final;

private:
    explicit WebSocket(ScriptExecutionContext&);

    void dispatchErrorEventIfNeeded();

    void contextDestroyed() final;

    // ActiveDOMObject.
    void suspend(ReasonForSuspension) final;
    void resume() final;
    void stop() final;

    enum EventTargetInterfaceType eventTargetInterface() const final;

    void refEventTarget() final { ref(); }
    void derefEventTarget() final { deref(); }

    void didConnect() final;
    void didReceiveMessage(String&& message) final;
    void didReceiveBinaryData(Vector<uint8_t>&&) final;
    void didReceiveMessageError(String&& reason) final;
    void didUpdateBufferedAmount(unsigned bufferedAmount) final;
    void didStartClosingHandshake() final;
    void didClose(unsigned unhandledBufferedAmount, ClosingHandshakeCompletionStatus, unsigned short code, const String& reason) final;
    void didUpgradeURL() final;

    size_t getFramingOverhead(size_t payloadSize);

    void failAsynchronously();

    static Lock s_allActiveWebSocketsLock;
    RefPtr<ThreadableWebSocketChannel> m_channel;

    State m_state { CONNECTING };
    URL m_url;
    unsigned m_bufferedAmount { 0 };
    unsigned m_bufferedAmountAfterClose { 0 };
    BinaryType m_binaryType { BinaryType::Blob };
    String m_subprotocol;
    String m_extensions;

    bool m_dispatchedErrorEvent { false };
    RefPtr<PendingActivity<WebSocket>> m_pendingActivity;
};

} // namespace WebCore
