/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 16, 2024.
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

#include "ScriptExecutionContext.h"
#include "ThreadableWebSocketChannel.h"
#include "WebSocketChannelClient.h"
#include "WorkerThreadableWebSocketChannel.h"
#include <memory>
#include <wtf/Forward.h>
#include <wtf/ThreadSafeWeakPtr.h>
#include <wtf/Vector.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

class ScriptExecutionContext;
class WebSocketChannelClient;

class ThreadableWebSocketChannelClientWrapper : public ThreadSafeRefCountedAndCanMakeThreadSafeWeakPtr<ThreadableWebSocketChannelClientWrapper> {
public:
    static Ref<ThreadableWebSocketChannelClientWrapper> create(ScriptExecutionContext&, WebSocketChannelClient&);

    WorkerThreadableWebSocketChannel::Peer* peer() const;
    void didCreateWebSocketChannel(Ref<WorkerThreadableWebSocketChannel::Peer>&&);
    void clearPeer();

    bool failedWebSocketChannelCreation() const;
    void setFailedWebSocketChannelCreation();

    // Subprotocol and extensions will be available when didConnect() callback is invoked.
    String subprotocol() const;
    void setSubprotocol(const String&);
    String extensions() const;
    void setExtensions(const String&);

    void clearClient();

    void didConnect();
    void didReceiveMessage(String&& message);
    void didReceiveBinaryData(Vector<uint8_t>&&);
    void didUpdateBufferedAmount(unsigned bufferedAmount);
    void didStartClosingHandshake();
    void didClose(unsigned unhandledBufferedAmount, WebSocketChannelClient::ClosingHandshakeCompletionStatus, unsigned short code, const String& reason);
    void didReceiveMessageError(String&& reason);
    void didUpgradeURL();

    void suspend();
    void resume();

private:
    ThreadableWebSocketChannelClientWrapper(ScriptExecutionContext&, WebSocketChannelClient&);

    void processPendingTasks();

    WeakPtr<ScriptExecutionContext> m_context;
    ThreadSafeWeakPtr<WebSocketChannelClient> m_client;
    RefPtr<WorkerThreadableWebSocketChannel::Peer> m_peer;
    bool m_failedWebSocketChannelCreation;
    // ThreadSafeRefCounted must not have String member variables.
    Vector<UChar> m_subprotocol;
    Vector<UChar> m_extensions;
    bool m_suspended;
    Vector<std::unique_ptr<ScriptExecutionContext::Task>> m_pendingTasks;
};

} // namespace WebCore
