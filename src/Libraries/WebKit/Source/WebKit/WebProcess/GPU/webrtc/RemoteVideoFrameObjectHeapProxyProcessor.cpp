/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 8, 2024.
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
#include "RemoteVideoFrameObjectHeapProxy.h"

#if PLATFORM(COCOA) && ENABLE(GPU_PROCESS) && ENABLE(VIDEO)

#include "RemoteVideoFrameObjectHeap.h"
#include "RemoteVideoFrameObjectHeapMessages.h"
#include "RemoteVideoFrameObjectHeapProxyProcessorMessages.h"
#include "RemoteVideoFrameProxy.h"
#include "WebProcess.h"
#include <WebCore/NativeImage.h>
#include <WebCore/PixelBufferConformerCV.h>

namespace WebKit {

using namespace WebCore;

Ref<RemoteVideoFrameObjectHeapProxyProcessor> RemoteVideoFrameObjectHeapProxyProcessor::create(GPUProcessConnection& connection)
{
    auto processor = adoptRef(*new RemoteVideoFrameObjectHeapProxyProcessor(connection));
    processor->initialize();
    return processor;
}

RemoteVideoFrameObjectHeapProxyProcessor::RemoteVideoFrameObjectHeapProxyProcessor(GPUProcessConnection& connection)
    : m_connection(&connection.connection())
    , m_queue(WorkQueue::create("RemoteVideoFrameObjectHeapProxy"_s, WorkQueue::QOS::UserInteractive))
{
    connection.addClient(*this);
}

RemoteVideoFrameObjectHeapProxyProcessor::~RemoteVideoFrameObjectHeapProxyProcessor()
{
    ASSERT(isMainRunLoop());
    clearCallbacks();
}

void RemoteVideoFrameObjectHeapProxyProcessor::initialize()
{
    RefPtr<IPC::Connection> connection;
    {
        Locker lock(m_connectionLock);
        connection = m_connection;
    }

    connection->addWorkQueueMessageReceiver(Messages::RemoteVideoFrameObjectHeapProxyProcessor::messageReceiverName(), protectedQueue(), *this);
}

void RemoteVideoFrameObjectHeapProxyProcessor::gpuProcessConnectionDidClose(GPUProcessConnection& connection)
{
    m_sharedVideoFrameWriter.disable();
    {
        Locker lock(m_connectionLock);
        m_connection = nullptr;
    }
    connection.protectedConnection()->removeWorkQueueMessageReceiver(Messages::RemoteVideoFrameObjectHeapProxyProcessor::messageReceiverName());
    clearCallbacks();
}

void RemoteVideoFrameObjectHeapProxyProcessor::clearCallbacks()
{
    HashMap<RemoteVideoFrameIdentifier, Callback> callbacks;
    {
        Locker lock(m_callbacksLock);
        callbacks = std::exchange(m_callbacks, { });
    }

    protectedQueue()->dispatch([queue = m_queue, callbacks = WTFMove(callbacks)]() mutable {
        for (auto& callback : callbacks.values())
            callback({ });
    });
}

void RemoteVideoFrameObjectHeapProxyProcessor::setSharedVideoFrameSemaphore(IPC::Semaphore&& semaphore)
{
    m_sharedVideoFrameReader.setSemaphore(WTFMove(semaphore));
}

void RemoteVideoFrameObjectHeapProxyProcessor::setSharedVideoFrameMemory(SharedMemory::Handle&& handle)
{
    m_sharedVideoFrameReader.setSharedMemory(WTFMove(handle));
}

RemoteVideoFrameObjectHeapProxyProcessor::Callback RemoteVideoFrameObjectHeapProxyProcessor::takeCallback(RemoteVideoFrameIdentifier identifier)
{
    Locker lock(m_callbacksLock);
    return m_callbacks.take(identifier);
}

void RemoteVideoFrameObjectHeapProxyProcessor::newVideoFrameBuffer(RemoteVideoFrameIdentifier identifier, std::optional<SharedVideoFrame::Buffer>&& sharedVideoFrameBuffer)
{
    RetainPtr<CVPixelBufferRef> pixelBuffer;
    if (sharedVideoFrameBuffer)
        pixelBuffer = m_sharedVideoFrameReader.readBuffer(WTFMove(*sharedVideoFrameBuffer));
    if (auto callback = takeCallback(identifier))
        callback(WTFMove(pixelBuffer));
}

void RemoteVideoFrameObjectHeapProxyProcessor::getVideoFrameBuffer(const RemoteVideoFrameProxy& frame, bool canUseIOSurface, Callback&& callback)
{
    {
        Locker lock(m_callbacksLock);
        ASSERT(!m_callbacks.contains(frame.identifier()));
        m_callbacks.add(frame.identifier(), WTFMove(callback));
    }
    RefPtr<IPC::Connection> connection;
    {
        Locker lock(m_connectionLock);
        connection = m_connection;
    }
    if (!connection) {
        takeCallback(frame.identifier())(nullptr);
        return;
    }
    connection->send(Messages::RemoteVideoFrameObjectHeap::GetVideoFrameBuffer(frame.newReadReference(), canUseIOSurface), 0, IPC::SendOption::DispatchMessageEvenWhenWaitingForSyncReply);
}

void RemoteVideoFrameObjectHeapProxyProcessor::newConvertedVideoFrameBuffer(std::optional<SharedVideoFrame::Buffer>&& buffer)
{
    ASSERT(!m_convertedBuffer);
    if (buffer)
        m_convertedBuffer = m_sharedVideoFrameReader.readBuffer(WTFMove(*buffer));
    m_conversionSemaphore.signal();
}

RefPtr<NativeImage> RemoteVideoFrameObjectHeapProxyProcessor::getNativeImage(const WebCore::VideoFrame& videoFrame)
{
    Ref connection = WebProcess::singleton().ensureGPUProcessConnection().connection();

    if (m_sharedVideoFrameWriter.isDisabled())
        m_sharedVideoFrameWriter = { };

    auto frame = m_sharedVideoFrameWriter.write(videoFrame,
        [&](auto& semaphore) { connection->send(Messages::RemoteVideoFrameObjectHeap::SetSharedVideoFrameSemaphore { semaphore }, 0); },
        [&](auto&& handle) { connection->send(Messages::RemoteVideoFrameObjectHeap::SetSharedVideoFrameMemory { WTFMove(handle) }, 0); });
    if (!frame)
        return nullptr;

    auto sendResult = connection->sendSync(Messages::RemoteVideoFrameObjectHeap::ConvertFrameBuffer { WTFMove(*frame) }, 0, GPUProcessConnection::defaultTimeout);
    if (!sendResult.succeeded()) {
        m_sharedVideoFrameWriter.disable();
        return nullptr;
    }

    auto [destinationColorSpace] = sendResult.takeReplyOr(DestinationColorSpace { DestinationColorSpace::SRGB().platformColorSpace() });

    m_conversionSemaphore.wait();

    auto pixelBuffer = WTFMove(m_convertedBuffer);
    return pixelBuffer ? NativeImage::create(PixelBufferConformerCV::imageFrom32BGRAPixelBuffer(WTFMove(pixelBuffer), destinationColorSpace.platformColorSpace())) : nullptr;
}

}

#endif
