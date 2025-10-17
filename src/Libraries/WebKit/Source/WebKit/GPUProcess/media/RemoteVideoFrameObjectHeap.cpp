/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 25, 2024.
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
#include "RemoteVideoFrameObjectHeap.h"

#if ENABLE(GPU_PROCESS) && ENABLE(VIDEO)
#include "Logging.h"
#include "RemoteVideoFrameObjectHeapMessages.h"
#include "RemoteVideoFrameObjectHeapProxyProcessorMessages.h"
#include "RemoteVideoFrameProxy.h"
#include <wtf/MainThread.h>
#include <wtf/Scope.h>
#include <wtf/WorkQueue.h>

#if PLATFORM(COCOA)
#include <WebCore/ColorSpaceCG.h>
#include <WebCore/PixelBufferConformerCV.h>
#include <WebCore/VideoFrameCV.h>
#include <pal/cf/CoreMediaSoftLink.h>
#include <WebCore/CoreVideoSoftLink.h>
#endif

namespace WebKit {

using namespace WebCore;

static WorkQueue& remoteVideoFrameObjectHeapQueueSingleton()
{
    static NeverDestroyed queue = WorkQueue::create("org.webkit.RemoteVideoFrameObjectHeap"_s, WorkQueue::QOS::UserInteractive);
    return queue.get();
}

Ref<RemoteVideoFrameObjectHeap> RemoteVideoFrameObjectHeap::create(Ref<IPC::Connection>&& connection)
{
    Ref heap = adoptRef(*new RemoteVideoFrameObjectHeap(WTFMove(connection)));
    heap->protectedConnection()->addWorkQueueMessageReceiver(Messages::RemoteVideoFrameObjectHeap::messageReceiverName(), remoteVideoFrameObjectHeapQueueSingleton(), heap);
    return heap;
}

RemoteVideoFrameObjectHeap::RemoteVideoFrameObjectHeap(Ref<IPC::Connection>&& connection)
    : m_connection(WTFMove(connection))
{
}

RemoteVideoFrameObjectHeap::~RemoteVideoFrameObjectHeap()
{
    ASSERT(m_isClosed);
}

void RemoteVideoFrameObjectHeap::close()
{
    assertIsMainThread();

    if (m_isClosed)
        return;

    m_isClosed = true;
    protectedConnection()->removeWorkQueueMessageReceiver(Messages::RemoteVideoFrameObjectHeap::messageReceiverName());

#if PLATFORM(COCOA)
    m_sharedVideoFrameWriter.disable();
#endif

    // Clients might hold on to the ref after this happens. They should also stop themselves, but if they do not,
    // avoid big memory leaks by clearing the frames. The clients should fail gracefully (do nothing) in case they fail to look up
    // frames.
    m_heap.clear();
    // TODO: add can happen after stopping.
}

RemoteVideoFrameProxy::Properties RemoteVideoFrameObjectHeap::add(Ref<WebCore::VideoFrame>&& frame)
{
    auto newFrameReference = RemoteVideoFrameReference::generateForAdd();
    auto properties = RemoteVideoFrameProxy::properties(newFrameReference, frame);
    auto success = m_heap.add(newFrameReference, WTFMove(frame));
    ASSERT_UNUSED(success, success);
    return properties;
}

void RemoteVideoFrameObjectHeap::releaseVideoFrame(RemoteVideoFrameWriteReference&& write)
{
    assertIsCurrent(remoteVideoFrameObjectHeapQueueSingleton());
    m_heap.remove(WTFMove(write));
}

#if PLATFORM(COCOA)
void RemoteVideoFrameObjectHeap::getVideoFrameBuffer(RemoteVideoFrameReadReference&& read, bool canSendIOSurface)
{
    assertIsCurrent(remoteVideoFrameObjectHeapQueueSingleton());

    auto identifier = read.identifier();
    auto videoFrame = get(WTFMove(read));

    std::optional<SharedVideoFrame::Buffer> buffer;
    Ref connection = m_connection;

    if (videoFrame) {
        buffer = m_sharedVideoFrameWriter.writeBuffer(videoFrame->pixelBuffer(),
            [&](auto& semaphore) { connection->send(Messages::RemoteVideoFrameObjectHeapProxyProcessor::SetSharedVideoFrameSemaphore { semaphore }, 0); },
            [&](SharedMemory::Handle&& handle) { connection->send(Messages::RemoteVideoFrameObjectHeapProxyProcessor::SetSharedVideoFrameMemory { WTFMove(handle) }, 0); },
            canSendIOSurface);
        // FIXME: We should ASSERT(result) once we support enough pixel buffer types.
    }
    connection->send(Messages::RemoteVideoFrameObjectHeapProxyProcessor::NewVideoFrameBuffer { identifier, WTFMove(buffer) }, 0);
}

void RemoteVideoFrameObjectHeap::pixelBuffer(RemoteVideoFrameReadReference&& read, CompletionHandler<void(RetainPtr<CVPixelBufferRef>)>&& completionHandler)
{
    assertIsCurrent(remoteVideoFrameObjectHeapQueueSingleton());

    auto videoFrame = get(WTFMove(read));
    if (!videoFrame) {
        ASSERT_IS_TESTING_IPC();
        completionHandler(nullptr);
        return;
    }

    auto pixelBuffer = videoFrame->pixelBuffer();
    ASSERT(pixelBuffer);
    completionHandler(WTFMove(pixelBuffer));
}

void RemoteVideoFrameObjectHeap::convertFrameBuffer(SharedVideoFrame&& sharedVideoFrame, CompletionHandler<void(WebCore::DestinationColorSpace)>&& callback)
{
    DestinationColorSpace destinationColorSpace { DestinationColorSpace::SRGB().platformColorSpace() };
    auto scope = makeScopeExit([&callback, &destinationColorSpace] { callback(destinationColorSpace); });

    RefPtr<VideoFrame> frame;
    Ref connection = m_connection;

    if (std::holds_alternative<RemoteVideoFrameReadReference>(sharedVideoFrame.buffer))
        frame = get(WTFMove(std::get<RemoteVideoFrameReadReference>(sharedVideoFrame.buffer)));
    else
        frame = m_sharedVideoFrameReader.read(WTFMove(sharedVideoFrame));

    if (!frame) {
        connection->send(Messages::RemoteVideoFrameObjectHeapProxyProcessor::NewConvertedVideoFrameBuffer { { } }, 0);
        return;
    }

    RetainPtr<CVPixelBufferRef> buffer = frame->pixelBuffer();
    destinationColorSpace = DestinationColorSpace(createCGColorSpaceForCVPixelBuffer(buffer.get()));

    if (CVPixelBufferGetPixelFormatType(buffer.get()) != kCVPixelFormatType_32BGRA) {
        Locker locker { m_pixelBufferConformerLock };
        if (!m_pixelBufferConformer)
            m_pixelBufferConformer = createPixelConformer().moveToUniquePtr();

        auto convertedBuffer = m_pixelBufferConformer->convert(buffer.get());
        if (!convertedBuffer) {
            RELEASE_LOG_ERROR(WebRTC, "RemoteVideoFrameObjectHeap::convertFrameBuffer conformer failed");
            connection->send(Messages::RemoteVideoFrameObjectHeapProxyProcessor::NewConvertedVideoFrameBuffer { { } }, 0);
            return;
        }
        buffer = WTFMove(convertedBuffer);
    }

    bool canSendIOSurface = false;
    auto result = m_sharedVideoFrameWriter.writeBuffer(buffer.get(),
        [&](auto& semaphore) { connection->send(Messages::RemoteVideoFrameObjectHeapProxyProcessor::SetSharedVideoFrameSemaphore { semaphore }, 0); },
        [&](auto&& handle) { connection->send(Messages::RemoteVideoFrameObjectHeapProxyProcessor::SetSharedVideoFrameMemory { WTFMove(handle) }, 0); },
        canSendIOSurface);
    connection->send(Messages::RemoteVideoFrameObjectHeapProxyProcessor::NewConvertedVideoFrameBuffer { WTFMove(result) }, 0);
}

void RemoteVideoFrameObjectHeap::setSharedVideoFrameSemaphore(IPC::Semaphore&& semaphore)
{
    m_sharedVideoFrameReader.setSemaphore(WTFMove(semaphore));
}

void RemoteVideoFrameObjectHeap::setSharedVideoFrameMemory(SharedMemory::Handle&& handle)
{
    m_sharedVideoFrameReader.setSharedMemory(WTFMove(handle));
}

#endif

void RemoteVideoFrameObjectHeap::lowMemoryHandler()
{
#if PLATFORM(COCOA)
    Locker locker { m_pixelBufferConformerLock };
    m_pixelBufferConformer = nullptr;
#endif
}

}

#endif
