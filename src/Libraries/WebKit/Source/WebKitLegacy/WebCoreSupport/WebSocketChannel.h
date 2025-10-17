/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 20, 2021.
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

#include "SocketStreamHandleClient.h"
#include <WebCore/ExceptionCode.h>
#include <WebCore/FileReaderLoaderClient.h>
#include <WebCore/ThreadableWebSocketChannel.h>
#include <WebCore/Timer.h>
#include <WebCore/WebSocketChannelInspector.h>
#include <WebCore/WebSocketDeflateFramer.h>
#include <WebCore/WebSocketFrame.h>
#include <WebCore/WebSocketHandshake.h>
#include <wtf/Deque.h>
#include <wtf/Forward.h>
#include <wtf/ObjectIdentifier.h>
#include <wtf/RefCounted.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/TypeCasts.h>
#include <wtf/Vector.h>
#include <wtf/text/CString.h>

namespace WebCore {

class Blob;
class Document;
class FileReaderLoader;
class ResourceRequest;
class ResourceResponse;
class SocketProvider;
class SocketStreamHandle;
class SocketStreamError;
class WebSocketChannelClient;
class WebSocketChannel;
using WebSocketChannelIdentifier = AtomicObjectIdentifier<WebSocketChannel>;

class WebSocketChannel final : public RefCounted<WebSocketChannel>, public SocketStreamHandleClient, public ThreadableWebSocketChannel, public FileReaderLoaderClient
{
    WTF_MAKE_TZONE_ALLOCATED(WebSocketChannel);
public:
    static Ref<WebSocketChannel> create(Document& document, WebSocketChannelClient& client, SocketProvider& provider) { return adoptRef(*new WebSocketChannel(document, client, provider)); }
    virtual ~WebSocketChannel();

    void send(std::span<const uint8_t> data);

    // ThreadableWebSocketChannel functions.
    ConnectStatus connect(const URL&, const String& protocol) final;
    String subprotocol() final;
    String extensions() final;
    void send(CString&&) final;
    void send(const JSC::ArrayBuffer&, unsigned byteOffset, unsigned byteLength) final;
    void send(Blob&) final;
    void close(int code, const String& reason) final; // Start closing handshake.
    void fail(String&& reason) final;
    void disconnect() final;

    void suspend() final;
    void resume() final;

    // SocketStreamHandleClient functions.
    void didOpenSocketStream(SocketStreamHandle&) final;
    void didCloseSocketStream(SocketStreamHandle&) final;
    void didReceiveSocketStreamData(SocketStreamHandle&, std::span<const uint8_t>) final;
    void didFailToReceiveSocketStreamData(SocketStreamHandle&) final;
    void didUpdateBufferedAmount(SocketStreamHandle&, size_t bufferedAmount) final;
    void didFailSocketStream(SocketStreamHandle&, const SocketStreamError&) final;

    // FileReaderLoaderClient functions.
    void didStartLoading() override;
    void didReceiveData() override;
    void didFinishLoading() override;
    void didFail(ExceptionCode errorCode) override;

    WebSocketChannelIdentifier progressIdentifier() const final { return m_progressIdentifier; }
    bool hasCreatedHandshake() const final { return !!m_handshake; }
    bool isConnected() const final { return m_handshake->mode() == WebSocketHandshake::Mode::Connected; }
    ResourceRequest clientHandshakeRequest(const CookieGetter&) const final;
    const ResourceResponse& serverHandshakeResponse() const final;

    using RefCounted<WebSocketChannel>::ref;
    using RefCounted<WebSocketChannel>::deref;

    Document* document();
    
private:
    WEBCORE_EXPORT WebSocketChannel(Document&, WebSocketChannelClient&, SocketProvider&);

    void refThreadableWebSocketChannel() override { ref(); }
    void derefThreadableWebSocketChannel() override { deref(); }

    bool appendToBuffer(std::span<const uint8_t>);
    void skipBuffer(size_t len);
    bool processBuffer();
    void resumeTimerFired();
    void startClosingHandshake(int code, const String& reason);
    void closingTimerFired();

    bool processFrame();

    // It is allowed to send a Blob as a binary frame if hybi-10 protocol is in use. Sending a Blob
    // can be delayed because it must be read asynchronously. Other types of data (String or
    // ArrayBuffer) may also be blocked by preceding sending request of a Blob.
    //
    // To address this situation, messages to be sent need to be stored in a queue. Whenever a new
    // data frame is going to be sent, it first must go to the queue. Items in the queue are processed
    // in the order they were put into the queue. Sending request of a Blob blocks further processing
    // until the Blob is completely read and sent to the socket stream.
    enum QueuedFrameType {
        QueuedFrameTypeString,
        QueuedFrameTypeVector,
        QueuedFrameTypeBlob
    };
    struct QueuedFrame {
        WTF_MAKE_STRUCT_FAST_ALLOCATED;

        WebSocketFrame::OpCode opCode;
        QueuedFrameType frameType;
        // Only one of the following items is used, according to the value of frameType.
        CString stringData;
        Vector<uint8_t> vectorData;
        RefPtr<Blob> blobData;
    };
    void enqueueTextFrame(CString&&);
    void enqueueRawFrame(WebSocketFrame::OpCode, std::span<const uint8_t> data);
    void enqueueBlobFrame(WebSocketFrame::OpCode, Blob&);

    void processOutgoingFrameQueue();
    void abortOutgoingFrameQueue();

    enum OutgoingFrameQueueStatus {
        // It is allowed to put a new item into the queue.
        OutgoingFrameQueueOpen,
        // Close frame has already been put into the queue but may not have been sent yet;
        // m_handle->close() will be called as soon as the queue is cleared. It is not
        // allowed to put a new item into the queue.
        OutgoingFrameQueueClosing,
        // Close frame has been sent or the queue was aborted. It is not allowed to put
        // a new item to the queue.
        OutgoingFrameQueueClosed
    };

    // If you are going to send a hybi-10 frame, you need to use the outgoing frame queue
    // instead of call sendFrame() directly.
    void sendFrame(WebSocketFrame::OpCode, std::span<const uint8_t> data, Function<void(bool)> completionHandler);

    enum BlobLoaderStatus {
        BlobLoaderNotStarted,
        BlobLoaderStarted,
        BlobLoaderFinished,
        BlobLoaderFailed
    };

    RefPtr<WebSocketChannelClient> protectedClient() const;

    WeakPtr<Document, WeakPtrImplWithEventTargetData> m_document;
    ThreadSafeWeakPtr<WebSocketChannelClient> m_client;
    std::unique_ptr<WebSocketHandshake> m_handshake;
    RefPtr<SocketStreamHandle> m_handle;
    Vector<uint8_t> m_buffer;

    Timer m_resumeTimer;
    bool m_suspended { false };
    bool m_closing { false };
    bool m_receivedClosingHandshake { false };
    bool m_allowCookies { true };
    Timer m_closingTimer;
    bool m_closed { false };
    bool m_shouldDiscardReceivedData { false };
    unsigned m_unhandledBufferedAmount { 0 };

    WebSocketChannelIdentifier m_progressIdentifier;

    // Private members only for hybi-10 protocol.
    bool m_hasContinuousFrame { false };
    WebSocketFrame::OpCode m_continuousFrameOpCode;
    Vector<uint8_t> m_continuousFrameData;
    unsigned short m_closeEventCode { CloseEventCodeAbnormalClosure };
    String m_closeEventReason;

    Deque<std::unique_ptr<QueuedFrame>> m_outgoingFrameQueue;
    OutgoingFrameQueueStatus m_outgoingFrameQueueStatus { OutgoingFrameQueueOpen };

    // FIXME: Load two or more Blobs simultaneously for better performance.
    std::unique_ptr<FileReaderLoader> m_blobLoader;
    BlobLoaderStatus m_blobLoaderStatus { BlobLoaderNotStarted };

    WebSocketDeflateFramer m_deflateFramer;
    Ref<SocketProvider> m_socketProvider;
};

} // namespace WebCore
