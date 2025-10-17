/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 6, 2023.
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

#if ENABLE(GPU_PROCESS) && PLATFORM(COCOA) && ENABLE(VIDEO)

#include "Connection.h"
#include "GPUProcessConnection.h"
#include "MessageReceiver.h"
#include "RemoteVideoFrameIdentifier.h"
#include "SharedVideoFrame.h"
#include "WorkQueueMessageReceiver.h"
#include <wtf/Function.h>
#include <wtf/HashMap.h>
#include <wtf/Lock.h>
#include <wtf/WorkQueue.h>
#include <wtf/threads/BinarySemaphore.h>

using CVPixelBufferPoolRef = struct __CVPixelBufferPool*;

namespace IPC {
class Connection;
class Decoder;
}

namespace WebCore {
class NativeImage;
class VideoFrame;
}

namespace WebKit {

class RemoteVideoFrameProxy;

class RemoteVideoFrameObjectHeapProxyProcessor : public IPC::WorkQueueMessageReceiver, public GPUProcessConnection::Client {
public:
    static Ref<RemoteVideoFrameObjectHeapProxyProcessor> create(GPUProcessConnection&);
    ~RemoteVideoFrameObjectHeapProxyProcessor();

    using Callback = Function<void(RetainPtr<CVPixelBufferRef>&&)>;
    void getVideoFrameBuffer(const RemoteVideoFrameProxy&, bool canUseIOSurfce, Callback&&);
    RefPtr<WebCore::NativeImage> getNativeImage(const WebCore::VideoFrame&);

    ThreadSafeWeakPtrControlBlock& controlBlock() const final { return IPC::WorkQueueMessageReceiver::controlBlock(); }
    size_t weakRefCount() const final { return IPC::WorkQueueMessageReceiver::weakRefCount(); }

    void ref() const final { IPC::WorkQueueMessageReceiver::ref(); }
    void deref() const final { IPC::WorkQueueMessageReceiver::deref(); }

private:
    explicit RemoteVideoFrameObjectHeapProxyProcessor(GPUProcessConnection&);
    void initialize();

    void didReceiveMessage(IPC::Connection&, IPC::Decoder&) final;

    // Messages
    void setSharedVideoFrameSemaphore(IPC::Semaphore&&);
    void setSharedVideoFrameMemory(WebCore::SharedMemory::Handle&&);
    void newVideoFrameBuffer(RemoteVideoFrameIdentifier, std::optional<SharedVideoFrame::Buffer>&&);
    void newConvertedVideoFrameBuffer(std::optional<SharedVideoFrame::Buffer>&&);

    // GPUProcessConnection::Client
    void gpuProcessConnectionDidClose(GPUProcessConnection&);

    void clearCallbacks();
    Callback takeCallback(RemoteVideoFrameIdentifier);

    Ref<WorkQueue> protectedQueue() const { return m_queue; }

    Lock m_connectionLock;
    RefPtr<IPC::Connection> m_connection WTF_GUARDED_BY_LOCK(m_connectionLock);
    Lock m_callbacksLock;
    HashMap<RemoteVideoFrameIdentifier, Callback> m_callbacks WTF_GUARDED_BY_LOCK(m_callbacksLock);
    Ref<WorkQueue> m_queue;
    SharedVideoFrameReader m_sharedVideoFrameReader;

    SharedVideoFrameWriter m_sharedVideoFrameWriter;
    RetainPtr<CVPixelBufferRef> m_convertedBuffer;
    BinarySemaphore m_conversionSemaphore;
};

} // namespace WebKit

#endif
