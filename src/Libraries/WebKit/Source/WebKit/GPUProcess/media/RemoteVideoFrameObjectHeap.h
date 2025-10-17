/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 9, 2024.
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

#if ENABLE(GPU_PROCESS) && ENABLE(VIDEO)
#include "Connection.h"
#include "RemoteVideoFrameProxy.h"
#include "ThreadSafeObjectHeap.h"
#include "WorkQueueMessageReceiver.h"

#if PLATFORM(COCOA)
#include "SharedVideoFrame.h"
#endif

namespace WebCore {
class DestinationColorSpace;
class PixelBufferConformerCV;
class VideoFrame;
}

namespace WebKit {

// Holds references to all VideoFrame instances that are mapped from GPU process to Web process.
class RemoteVideoFrameObjectHeap final : public IPC::WorkQueueMessageReceiver {
public:
    static Ref<RemoteVideoFrameObjectHeap> create(Ref<IPC::Connection>&&);
    ~RemoteVideoFrameObjectHeap();

    void ref() const final { IPC::WorkQueueMessageReceiver::ref(); }
    void deref() const final { IPC::WorkQueueMessageReceiver::deref(); }

    void close();
    RemoteVideoFrameProxy::Properties add(Ref<WebCore::VideoFrame>&&);
    RefPtr<WebCore::VideoFrame> get(RemoteVideoFrameReadReference&&);

    void stopListeningForIPC(Ref<RemoteVideoFrameObjectHeap>&&) { close(); }
    void lowMemoryHandler();

private:
    explicit RemoteVideoFrameObjectHeap(Ref<IPC::Connection>&&);

    Ref<IPC::Connection> protectedConnection() const { return m_connection; }

    // IPC::MessageReceiver overrides.
    void didReceiveMessage(IPC::Connection&, IPC::Decoder&) final;
    bool didReceiveSyncMessage(IPC::Connection&, IPC::Decoder&, UniqueRef<IPC::Encoder>&) final;

    // Messages.
    void releaseVideoFrame(RemoteVideoFrameWriteReference&&);
#if PLATFORM(COCOA)
    void getVideoFrameBuffer(RemoteVideoFrameReadReference&&, bool canSendIOSurface);
    void pixelBuffer(RemoteVideoFrameReadReference&&, CompletionHandler<void(RetainPtr<CVPixelBufferRef>)>&&);
    void convertFrameBuffer(SharedVideoFrame&&, CompletionHandler<void(WebCore::DestinationColorSpace)>&&);
    void setSharedVideoFrameSemaphore(IPC::Semaphore&&);
    void setSharedVideoFrameMemory(WebCore::SharedMemory::Handle&&);

    UniqueRef<WebCore::PixelBufferConformerCV> createPixelConformer();
#endif

    const Ref<IPC::Connection> m_connection;
    IPC::ThreadSafeObjectHeap<RemoteVideoFrameIdentifier, RefPtr<WebCore::VideoFrame>> m_heap;
#if PLATFORM(COCOA)
    SharedVideoFrameWriter m_sharedVideoFrameWriter;
    SharedVideoFrameReader m_sharedVideoFrameReader;
    Lock m_pixelBufferConformerLock;
    std::unique_ptr<WebCore::PixelBufferConformerCV> m_pixelBufferConformer WTF_GUARDED_BY_LOCK(m_pixelBufferConformerLock);
#endif
    bool m_isClosed { false };
};

}
#endif
