/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 17, 2023.
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
#include "WorkerThreadableWebSocketChannel.h"

#include "Blob.h"
#include "Document.h"
#include "FrameDestructionObserverInlines.h"
#include "LocalFrame.h"
#include "MixedContentChecker.h"
#include "ScriptExecutionContext.h"
#include "SocketProvider.h"
#include "ThreadableWebSocketChannelClientWrapper.h"
#include "WebSocketChannelClient.h"
#include "WorkerGlobalScope.h"
#include "WorkerLoaderProxy.h"
#include "WorkerRunLoop.h"
#include "WorkerThread.h"
#include <JavaScriptCore/ArrayBuffer.h>
#include <wtf/MainThread.h>
#include <wtf/StdLibExtras.h>
#include <wtf/TZoneMallocInlines.h>
#include <wtf/text/MakeString.h>
#include <wtf/text/WTFString.h>
#include <wtf/threads/BinarySemaphore.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(WorkerThreadableWebSocketChannel);

WorkerThreadableWebSocketChannel::WorkerThreadableWebSocketChannel(WorkerGlobalScope& context, WebSocketChannelClient& client, const String& taskMode, SocketProvider& provider)
    : m_workerGlobalScope(context)
    , m_workerClientWrapper(ThreadableWebSocketChannelClientWrapper::create(context, client))
    , m_bridge(Bridge::create(m_workerClientWrapper.copyRef(), m_workerGlobalScope.copyRef(), taskMode, provider))
    , m_socketProvider(provider)
    , m_progressIdentifier(WebSocketChannelIdentifier::generate())
{
    m_bridge->initialize(context);
}

WorkerThreadableWebSocketChannel::~WorkerThreadableWebSocketChannel()
{
    if (RefPtr bridge = m_bridge)
        bridge->disconnect();
}

WorkerThreadableWebSocketChannel::ConnectStatus WorkerThreadableWebSocketChannel::connect(const URL& url, const String& protocol)
{
    if (RefPtr bridge = m_bridge)
        bridge->connect(url, protocol);
    // connect is called asynchronously, so we do not have any possibility for synchronous errors.
    return ConnectStatus::OK;
}

String WorkerThreadableWebSocketChannel::subprotocol()
{
    return m_workerClientWrapper->subprotocol();
}

String WorkerThreadableWebSocketChannel::extensions()
{
    return m_workerClientWrapper->extensions();
}

void WorkerThreadableWebSocketChannel::send(CString&& message)
{
    if (RefPtr bridge = m_bridge)
        bridge->send(WTFMove(message));
}

void WorkerThreadableWebSocketChannel::send(const ArrayBuffer& binaryData, unsigned byteOffset, unsigned byteLength)
{
    if (RefPtr bridge = m_bridge)
        bridge->send(binaryData, byteOffset, byteLength);
}

void WorkerThreadableWebSocketChannel::send(Blob& binaryData)
{
    if (RefPtr bridge = m_bridge)
        bridge->send(binaryData);
}

void WorkerThreadableWebSocketChannel::close(int code, const String& reason)
{
    if (RefPtr bridge = m_bridge)
        bridge->close(code, reason);
}

void WorkerThreadableWebSocketChannel::fail(String&& reason)
{
    if (RefPtr bridge = m_bridge)
        bridge->fail(WTFMove(reason));
}

void WorkerThreadableWebSocketChannel::disconnect()
{
    std::exchange(m_bridge, nullptr)->disconnect();
}

void WorkerThreadableWebSocketChannel::suspend()
{
    m_workerClientWrapper->suspend();
    if (RefPtr bridge = m_bridge)
        bridge->suspend();
}

void WorkerThreadableWebSocketChannel::resume()
{
    m_workerClientWrapper->resume();
    if (RefPtr bridge = m_bridge)
        bridge->resume();
}

WTF_MAKE_TZONE_ALLOCATED_IMPL(WorkerThreadableWebSocketChannel::Peer);

WorkerThreadableWebSocketChannel::Peer::Peer(Ref<ThreadableWebSocketChannelClientWrapper>&& clientWrapper, ScriptExecutionContext& context, ScriptExecutionContextIdentifier workerContextIdentifier, const String& taskMode, SocketProvider& provider)
    : m_workerClientWrapper(clientWrapper.ptr())
    , m_mainWebSocketChannel(ThreadableWebSocketChannel::create(downcast<Document>(context), *this, provider))
    , m_taskMode(taskMode)
    , m_workerContextIdentifier(workerContextIdentifier)
{
    ASSERT(isMainThread());
}

Ref<WorkerThreadableWebSocketChannel::Peer> WorkerThreadableWebSocketChannel::Peer::create(Ref<ThreadableWebSocketChannelClientWrapper>&& clientWrapper, ScriptExecutionContext& context, ScriptExecutionContextIdentifier workerContextIdentifier, const String& taskMode, SocketProvider& provider)
{
    return adoptRef(*new Peer(WTFMove(clientWrapper), context, workerContextIdentifier, taskMode, provider));
}

WorkerThreadableWebSocketChannel::Peer::~Peer()
{
    ASSERT(isMainThread());
    if (RefPtr channel = m_mainWebSocketChannel)
        channel->disconnect();
}

WorkerThreadableWebSocketChannel::ConnectStatus WorkerThreadableWebSocketChannel::Peer::connect(const URL& url, const String& protocol)
{
    ASSERT(isMainThread());
    RefPtr channel = m_mainWebSocketChannel;
    if (!channel)
        return WorkerThreadableWebSocketChannel::ConnectStatus::KO;

    return channel->connect(url, protocol);
}

void WorkerThreadableWebSocketChannel::Peer::send(CString&& message)
{
    ASSERT(isMainThread());
    if (RefPtr channel = m_mainWebSocketChannel)
        channel->send(WTFMove(message));
}

void WorkerThreadableWebSocketChannel::Peer::send(const ArrayBuffer& binaryData)
{
    ASSERT(isMainThread());
    if (RefPtr channel = m_mainWebSocketChannel)
        channel->send(binaryData, 0, binaryData.byteLength());
}

void WorkerThreadableWebSocketChannel::Peer::send(Blob& binaryData)
{
    ASSERT(isMainThread());
    if (RefPtr channel = m_mainWebSocketChannel)
        m_mainWebSocketChannel->send(binaryData);
}

void WorkerThreadableWebSocketChannel::Peer::close(int code, const String& reason)
{
    ASSERT(isMainThread());
    if (RefPtr channel = m_mainWebSocketChannel)
        channel->close(code, reason);
}

void WorkerThreadableWebSocketChannel::Peer::fail(String&& reason)
{
    ASSERT(isMainThread());
    if (RefPtr channel = m_mainWebSocketChannel)
        channel->fail(WTFMove(reason));
}

void WorkerThreadableWebSocketChannel::Peer::disconnect()
{
    ASSERT(isMainThread());
    if (!m_mainWebSocketChannel)
        return;
    std::exchange(m_mainWebSocketChannel, nullptr)->disconnect();
}

void WorkerThreadableWebSocketChannel::Peer::suspend()
{
    ASSERT(isMainThread());
    if (RefPtr channel = m_mainWebSocketChannel)
        channel->suspend();
}

void WorkerThreadableWebSocketChannel::Peer::resume()
{
    ASSERT(isMainThread());
    if (RefPtr channel = m_mainWebSocketChannel)
        channel->resume();
}

void WorkerThreadableWebSocketChannel::Peer::didConnect()
{
    ASSERT(isMainThread());

    RefPtr workerClientWrapper = m_workerClientWrapper.get();
    if (!workerClientWrapper)
        return;

    String subprotocol = m_mainWebSocketChannel->subprotocol();
    String extensions = m_mainWebSocketChannel->extensions();
    ScriptExecutionContext::postTaskForModeToWorkerOrWorklet(m_workerContextIdentifier, [workerClientWrapper = workerClientWrapper.releaseNonNull(), subprotocol = WTFMove(subprotocol).isolatedCopy(), extensions = WTFMove(extensions).isolatedCopy()](ScriptExecutionContext& context) mutable {
        ASSERT_UNUSED(context, context.isWorkerGlobalScope() || context.isWorkletGlobalScope());
        workerClientWrapper->setSubprotocol(subprotocol);
        workerClientWrapper->setExtensions(extensions);
        workerClientWrapper->didConnect();
    }, m_taskMode);
}

void WorkerThreadableWebSocketChannel::Peer::didReceiveMessage(String&& message)
{
    ASSERT(isMainThread());

    RefPtr workerClientWrapper = m_workerClientWrapper.get();
    if (!workerClientWrapper)
        return;

    ScriptExecutionContext::postTaskForModeToWorkerOrWorklet(m_workerContextIdentifier, [workerClientWrapper = workerClientWrapper.releaseNonNull(), message = WTFMove(message).isolatedCopy()](ScriptExecutionContext& context) mutable {
        ASSERT_UNUSED(context, context.isWorkerGlobalScope() || context.isWorkletGlobalScope());
        workerClientWrapper->didReceiveMessage(WTFMove(message));
    }, m_taskMode);
}

void WorkerThreadableWebSocketChannel::Peer::didReceiveBinaryData(Vector<uint8_t>&& binaryData)
{
    ASSERT(isMainThread());

    RefPtr workerClientWrapper = m_workerClientWrapper.get();
    if (!workerClientWrapper)
        return;

    ScriptExecutionContext::postTaskForModeToWorkerOrWorklet(m_workerContextIdentifier, [workerClientWrapper = workerClientWrapper.releaseNonNull(), binaryData = WTFMove(binaryData)](ScriptExecutionContext& context) mutable {
        ASSERT_UNUSED(context, context.isWorkerGlobalScope() || context.isWorkletGlobalScope());
        workerClientWrapper->didReceiveBinaryData(WTFMove(binaryData));
    }, m_taskMode);
}

void WorkerThreadableWebSocketChannel::Peer::didUpdateBufferedAmount(unsigned bufferedAmount)
{
    ASSERT(isMainThread());

    RefPtr workerClientWrapper = m_workerClientWrapper.get();
    if (!workerClientWrapper)
        return;

    ScriptExecutionContext::postTaskForModeToWorkerOrWorklet(m_workerContextIdentifier, [workerClientWrapper = workerClientWrapper.releaseNonNull(), bufferedAmount](ScriptExecutionContext& context) mutable {
        ASSERT_UNUSED(context, context.isWorkerGlobalScope() || context.isWorkletGlobalScope());
        workerClientWrapper->didUpdateBufferedAmount(bufferedAmount);
    }, m_taskMode);
}

void WorkerThreadableWebSocketChannel::Peer::didStartClosingHandshake()
{
    ASSERT(isMainThread());

    RefPtr workerClientWrapper = m_workerClientWrapper.get();
    if (!workerClientWrapper)
        return;

    ScriptExecutionContext::postTaskForModeToWorkerOrWorklet(m_workerContextIdentifier, [workerClientWrapper = workerClientWrapper.releaseNonNull()](ScriptExecutionContext& context) mutable {
        ASSERT_UNUSED(context, context.isWorkerGlobalScope() || context.isWorkletGlobalScope());
        workerClientWrapper->didStartClosingHandshake();
    }, m_taskMode);
}

void WorkerThreadableWebSocketChannel::Peer::didClose(unsigned unhandledBufferedAmount, ClosingHandshakeCompletionStatus closingHandshakeCompletion, unsigned short code, const String& reason)
{
    ASSERT(isMainThread());
    m_mainWebSocketChannel = nullptr;

    RefPtr workerClientWrapper = m_workerClientWrapper.get();
    if (!workerClientWrapper)
        return;

    ScriptExecutionContext::postTaskForModeToWorkerOrWorklet(m_workerContextIdentifier, [workerClientWrapper = workerClientWrapper.releaseNonNull(), unhandledBufferedAmount, closingHandshakeCompletion, code, reason = reason.isolatedCopy()](ScriptExecutionContext& context) mutable {
        ASSERT_UNUSED(context, context.isWorkerGlobalScope() || context.isWorkletGlobalScope());
        workerClientWrapper->didClose(unhandledBufferedAmount, closingHandshakeCompletion, code, reason);
    }, m_taskMode);
}

void WorkerThreadableWebSocketChannel::Peer::didReceiveMessageError(String&& reason)
{
    ASSERT(isMainThread());

    RefPtr workerClientWrapper = m_workerClientWrapper.get();
    if (!workerClientWrapper)
        return;

    ScriptExecutionContext::postTaskForModeToWorkerOrWorklet(m_workerContextIdentifier, [workerClientWrapper = workerClientWrapper.releaseNonNull(), reason = WTFMove(reason).isolatedCopy()](ScriptExecutionContext& context) mutable {
        ASSERT_UNUSED(context, context.isWorkerGlobalScope() || context.isWorkletGlobalScope());
        workerClientWrapper->didReceiveMessageError(WTFMove(reason));
    }, m_taskMode);
}

void WorkerThreadableWebSocketChannel::Peer::didUpgradeURL()
{
    ASSERT(isMainThread());

    RefPtr workerClientWrapper = m_workerClientWrapper.get();
    if (!workerClientWrapper)
        return;
    
    ScriptExecutionContext::postTaskForModeToWorkerOrWorklet(m_workerContextIdentifier, [workerClientWrapper = workerClientWrapper.releaseNonNull()](ScriptExecutionContext& context) mutable {
        ASSERT_UNUSED(context, context.isWorkerGlobalScope() || context.isWorkletGlobalScope());
        workerClientWrapper->didUpgradeURL();
    }, m_taskMode);
}

WorkerThreadableWebSocketChannel::Bridge::Bridge(Ref<ThreadableWebSocketChannelClientWrapper>&& workerClientWrapper, Ref<WorkerGlobalScope>&& workerGlobalScope, const String& taskMode, Ref<SocketProvider>&& socketProvider)
    : m_workerClientWrapper(WTFMove(workerClientWrapper))
    , m_workerGlobalScope(WTFMove(workerGlobalScope))
    , m_loaderProxy(*m_workerGlobalScope->thread().workerLoaderProxy())
    , m_taskMode(taskMode)
    , m_socketProvider(WTFMove(socketProvider))
{
}

WorkerThreadableWebSocketChannel::Bridge::~Bridge()
{
    disconnect();
}

auto WorkerThreadableWebSocketChannel::Bridge::mainThreadInitialize(ScriptExecutionContext& context, WorkerThread& workerThread, ScriptExecutionContextIdentifier workerContextIdentifier, Ref<ThreadableWebSocketChannelClientWrapper>&& clientWrapper, const String& taskMode, Ref<SocketProvider>&& provider) -> RefPtr<Peer>
{
    ASSERT(isMainThread());
    ASSERT(context.isDocument());

    auto& workerRunLoop = workerThread.runLoop();
    if (workerRunLoop.terminated())
        return nullptr;

    return Peer::create(clientWrapper.copyRef(), context, workerContextIdentifier, taskMode, WTFMove(provider));
}

void WorkerThreadableWebSocketChannel::Bridge::initialize(WorkerGlobalScope& scope)
{
    ASSERT(!m_peer.get());
    Ref<Bridge> protectedThis(*this);

    RefPtr<Peer> peer;

    BinarySemaphore semaphore;
    m_loaderProxy.postTaskToLoader([&semaphore, &peer, workerThread = Ref { scope.thread() }, workerContextIdentifier = scope.identifier(), workerClientWrapper = m_workerClientWrapper, taskMode = m_taskMode.isolatedCopy(), provider = m_socketProvider](ScriptExecutionContext& context) mutable {
        peer = mainThreadInitialize(context, workerThread.get(), workerContextIdentifier, WTFMove(workerClientWrapper), taskMode, WTFMove(provider));
        semaphore.signal();
    });
    semaphore.wait();

    if (peer)
        m_workerClientWrapper->didCreateWebSocketChannel(peer.releaseNonNull());
    m_peer = m_workerClientWrapper->peer();
}

void WorkerThreadableWebSocketChannel::Bridge::connect(const URL& url, const String& protocol)
{
    RefPtr peer = m_peer.get();
    if (!peer)
        return;

    m_loaderProxy.postTaskToLoader([peer = peer.releaseNonNull(), url = url.isolatedCopy(), protocol = protocol.isolatedCopy()](ScriptExecutionContext& context) {
        ASSERT(isMainThread());

        auto& document = downcast<Document>(context);
        
        if (RefPtr frame = document.frame(); frame && MixedContentChecker::shouldBlockRequestForRunnableContent(*frame, document.securityOrigin(), url, MixedContentChecker::ShouldLogWarning::No)) {
            peer->fail(makeString("The page at "_s, document.url().stringCenterEllipsizedToLength(), " was blocked from connecting insecurely to "_s, url.stringCenterEllipsizedToLength(), " either because the protocol is insecure or the page is embedded from an insecure page."_s));
            return;
        }

        if (peer->connect(url, protocol) == ThreadableWebSocketChannel::ConnectStatus::KO)
            peer->didReceiveMessageError(String { });
    });
}

void WorkerThreadableWebSocketChannel::Bridge::send(CString&& message)
{
    RefPtr peer = m_peer.get();
    if (!peer)
        return;

    m_loaderProxy.postTaskToLoader([peer = peer.releaseNonNull(), message = WTFMove(message)](ScriptExecutionContext& context) mutable {
        ASSERT(isMainThread());
        ASSERT_UNUSED(context, context.isDocument());

        peer->send(WTFMove(message));
    });
}

void WorkerThreadableWebSocketChannel::Bridge::send(const ArrayBuffer& binaryData, unsigned byteOffset, unsigned byteLength)
{
    RefPtr peer = m_peer.get();
    if (!peer)
        return;

    // ArrayBuffer isn't thread-safe, hence the content of ArrayBuffer is copied into Vector<uint8_t>.
    Vector<uint8_t> data(byteLength);
    if (binaryData.byteLength())
        memcpySpan(data.mutableSpan(), binaryData.span().subspan(byteOffset, byteLength));

    m_loaderProxy.postTaskToLoader([peer = peer.releaseNonNull(), data = WTFMove(data)](ScriptExecutionContext& context) {
        ASSERT(isMainThread());
        ASSERT_UNUSED(context, context.isDocument());

        auto arrayBuffer = ArrayBuffer::create(data.span());
        peer->send(arrayBuffer);
    });
}

void WorkerThreadableWebSocketChannel::Bridge::send(Blob& binaryData)
{
    RefPtr peer = m_peer.get();
    if (!peer)
        return;

    m_loaderProxy.postTaskToLoader([peer = peer.releaseNonNull(), url = binaryData.url().isolatedCopy(), type = binaryData.type().isolatedCopy(), size = binaryData.size(), memoryCost = binaryData.memoryCost()](ScriptExecutionContext& context) {
        ASSERT(isMainThread());
        ASSERT_UNUSED(context, context.isDocument());

        peer->send(Blob::deserialize(&context, url, type, size, memoryCost, { }));
    });
}

void WorkerThreadableWebSocketChannel::Bridge::close(int code, const String& reason)
{
    RefPtr peer = m_peer.get();
    if (!peer)
        return;

    m_loaderProxy.postTaskToLoader([peer = peer.releaseNonNull(), code, reason = reason.isolatedCopy()](ScriptExecutionContext& context) {
        ASSERT(isMainThread());
        ASSERT_UNUSED(context, context.isDocument());

        peer->close(code, reason);
    });
}

void WorkerThreadableWebSocketChannel::Bridge::fail(String&& reason)
{
    RefPtr peer = m_peer.get();
    if (!peer)
        return;

    m_loaderProxy.postTaskToLoader([peer = peer.releaseNonNull(), reason = WTFMove(reason).isolatedCopy()](ScriptExecutionContext& context) mutable {
        ASSERT(isMainThread());
        ASSERT_UNUSED(context, context.isDocument());

        peer->fail(WTFMove(reason));
    });
}

void WorkerThreadableWebSocketChannel::Bridge::disconnect()
{
    clearClientWrapper();
    m_peer = nullptr;
    m_workerGlobalScope = nullptr;
}

void WorkerThreadableWebSocketChannel::Bridge::suspend()
{
    RefPtr peer = m_peer.get();
    if (!peer)
        return;

    m_loaderProxy.postTaskToLoader([peer = peer.releaseNonNull()](ScriptExecutionContext& context) {
        ASSERT(isMainThread());
        ASSERT_UNUSED(context, context.isDocument());

        peer->suspend();
    });
}

void WorkerThreadableWebSocketChannel::Bridge::resume()
{
    RefPtr peer = m_peer.get();
    if (!peer)
        return;

    m_loaderProxy.postTaskToLoader([peer = peer.releaseNonNull()](ScriptExecutionContext& context) {
        ASSERT(isMainThread());
        ASSERT_UNUSED(context, context.isDocument());

        peer->resume();
    });
}

void WorkerThreadableWebSocketChannel::Bridge::clearClientWrapper()
{
    m_workerClientWrapper->clearClient();
}

} // namespace WebCore
