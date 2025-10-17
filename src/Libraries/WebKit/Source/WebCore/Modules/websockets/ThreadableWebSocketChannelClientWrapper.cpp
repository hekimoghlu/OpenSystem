/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 6, 2025.
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
#include "config.h"
#include "ThreadableWebSocketChannelClientWrapper.h"

#include "ScriptExecutionContext.h"
#include "WebSocketChannelClient.h"
#include <wtf/RefPtr.h>
#include <wtf/text/StringView.h>

namespace WebCore {

ThreadableWebSocketChannelClientWrapper::ThreadableWebSocketChannelClientWrapper(ScriptExecutionContext& context, WebSocketChannelClient& client)
    : m_context(context)
    , m_client(client)
    , m_failedWebSocketChannelCreation(false)
    , m_suspended(false)
{
}

Ref<ThreadableWebSocketChannelClientWrapper> ThreadableWebSocketChannelClientWrapper::create(ScriptExecutionContext& context, WebSocketChannelClient& client)
{
    return adoptRef(*new ThreadableWebSocketChannelClientWrapper(context, client));
}

WorkerThreadableWebSocketChannel::Peer* ThreadableWebSocketChannelClientWrapper::peer() const
{
    return m_peer.get();
}

void ThreadableWebSocketChannelClientWrapper::didCreateWebSocketChannel(Ref<WorkerThreadableWebSocketChannel::Peer>&& peer)
{
    m_peer = WTFMove(peer);
}

void ThreadableWebSocketChannelClientWrapper::clearPeer()
{
    m_peer = nullptr;
}

bool ThreadableWebSocketChannelClientWrapper::failedWebSocketChannelCreation() const
{
    return m_failedWebSocketChannelCreation;
}

void ThreadableWebSocketChannelClientWrapper::setFailedWebSocketChannelCreation()
{
    m_failedWebSocketChannelCreation = true;
}

String ThreadableWebSocketChannelClientWrapper::subprotocol() const
{
    if (m_subprotocol.isEmpty())
        return emptyString();
    return String(m_subprotocol);
}

void ThreadableWebSocketChannelClientWrapper::setSubprotocol(const String& subprotocol)
{
    unsigned length = subprotocol.length();
    m_subprotocol.resize(length);
    StringView(subprotocol).getCharacters(m_subprotocol.mutableSpan());
}

String ThreadableWebSocketChannelClientWrapper::extensions() const
{
    if (m_extensions.isEmpty())
        return emptyString();
    return String(m_extensions);
}

void ThreadableWebSocketChannelClientWrapper::setExtensions(const String& extensions)
{
    unsigned length = extensions.length();
    m_extensions.resize(length);
    StringView(extensions).getCharacters(m_extensions.mutableSpan());
}

void ThreadableWebSocketChannelClientWrapper::clearClient()
{
    m_client = nullptr;
}

void ThreadableWebSocketChannelClientWrapper::didConnect()
{
    m_pendingTasks.append(makeUnique<ScriptExecutionContext::Task>([this, protectedThis = Ref { *this }] (ScriptExecutionContext&) {
        if (RefPtr client = m_client.get())
            client->didConnect();
    }));

    if (!m_suspended)
        processPendingTasks();
}

void ThreadableWebSocketChannelClientWrapper::didReceiveMessage(String&& message)
{
    m_pendingTasks.append(makeUnique<ScriptExecutionContext::Task>([this, protectedThis = Ref { *this }, message = WTFMove(message).isolatedCopy()] (ScriptExecutionContext&) mutable {
        if (RefPtr client = m_client.get())
            client->didReceiveMessage(WTFMove(message));
    }));

    if (!m_suspended)
        processPendingTasks();
}

void ThreadableWebSocketChannelClientWrapper::didReceiveBinaryData(Vector<uint8_t>&& binaryData)
{
    m_pendingTasks.append(makeUnique<ScriptExecutionContext::Task>([this, protectedThis = Ref { *this }, binaryData = WTFMove(binaryData)] (ScriptExecutionContext&) mutable {
        if (RefPtr client = m_client.get())
            client->didReceiveBinaryData(WTFMove(binaryData));
    }));

    if (!m_suspended)
        processPendingTasks();
}

void ThreadableWebSocketChannelClientWrapper::didUpdateBufferedAmount(unsigned bufferedAmount)
{
    m_pendingTasks.append(makeUnique<ScriptExecutionContext::Task>([this, protectedThis = Ref { *this }, bufferedAmount] (ScriptExecutionContext&) {
        if (RefPtr client = m_client.get())
            client->didUpdateBufferedAmount(bufferedAmount);
    }));

    if (!m_suspended)
        processPendingTasks();
}

void ThreadableWebSocketChannelClientWrapper::didStartClosingHandshake()
{
    m_pendingTasks.append(makeUnique<ScriptExecutionContext::Task>([this, protectedThis = Ref { *this }] (ScriptExecutionContext&) {
        if (RefPtr client = m_client.get())
            client->didStartClosingHandshake();
    }));

    if (!m_suspended)
        processPendingTasks();
}

void ThreadableWebSocketChannelClientWrapper::didClose(unsigned unhandledBufferedAmount, WebSocketChannelClient::ClosingHandshakeCompletionStatus closingHandshakeCompletion, unsigned short code, const String& reason)
{
    m_pendingTasks.append(makeUnique<ScriptExecutionContext::Task>([this, protectedThis = Ref { *this }, unhandledBufferedAmount, closingHandshakeCompletion, code, reason = reason.isolatedCopy()] (ScriptExecutionContext&) {
        if (RefPtr client = m_client.get())
            client->didClose(unhandledBufferedAmount, closingHandshakeCompletion, code, reason);
    }));

    if (!m_suspended)
        processPendingTasks();
}

void ThreadableWebSocketChannelClientWrapper::didReceiveMessageError(String&& reason)
{
    m_pendingTasks.append(makeUnique<ScriptExecutionContext::Task>([this, protectedThis = Ref { *this }, reason = WTFMove(reason).isolatedCopy()] (ScriptExecutionContext&) mutable {
        if (RefPtr client = m_client.get())
            client->didReceiveMessageError(WTFMove(reason));
    }));

    if (!m_suspended)
        processPendingTasks();
}

void ThreadableWebSocketChannelClientWrapper::didUpgradeURL()
{
    m_pendingTasks.append(makeUnique<ScriptExecutionContext::Task>([this, protectedThis = Ref { *this }] (ScriptExecutionContext&) {
        if (RefPtr client = m_client.get())
            client->didUpgradeURL();
    }));
    
    if (!m_suspended)
        processPendingTasks();
}

void ThreadableWebSocketChannelClientWrapper::suspend()
{
    m_suspended = true;
}

void ThreadableWebSocketChannelClientWrapper::resume()
{
    m_suspended = false;
    processPendingTasks();
}

void ThreadableWebSocketChannelClientWrapper::processPendingTasks()
{
    if (m_suspended)
        return;

    Vector<std::unique_ptr<ScriptExecutionContext::Task>> pendingTasks = WTFMove(m_pendingTasks);
    Ref protectedContext = { *m_context };
    for (auto& task : pendingTasks)
        task->performTask(protectedContext.get());
}

} // namespace WebCore
