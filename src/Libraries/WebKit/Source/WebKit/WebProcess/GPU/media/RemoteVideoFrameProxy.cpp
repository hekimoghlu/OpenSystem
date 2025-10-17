/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 17, 2022.
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
#include "RemoteVideoFrameProxy.h"

#if ENABLE(GPU_PROCESS) && ENABLE(VIDEO)
#include "GPUConnectionToWebProcess.h"
#include "RemoteVideoFrameObjectHeapMessages.h"
#include "RemoteVideoFrameObjectHeapProxy.h"
#include <wtf/TZoneMallocInlines.h>

#if PLATFORM(COCOA)
#include "WebProcess.h"
#include <WebCore/CVUtilities.h>
#include <WebCore/RealtimeIncomingVideoSourceCocoa.h>
#include <WebCore/VideoFrameCV.h>
#include <wtf/MainThread.h>
#include <wtf/threads/BinarySemaphore.h>
#endif

namespace WebKit {

WTF_MAKE_TZONE_ALLOCATED_IMPL(RemoteVideoFrameProxy);

RemoteVideoFrameProxy::Properties RemoteVideoFrameProxy::properties(WebKit::RemoteVideoFrameReference reference, const WebCore::VideoFrame& videoFrame)
{
    return {
        WTFMove(reference),
        videoFrame.presentationTime(),
        videoFrame.isMirrored(),
        videoFrame.rotation(),
        expandedIntSize(videoFrame.presentationSize()),
        videoFrame.pixelFormat(),
        videoFrame.colorSpace()
    };
}

Ref<RemoteVideoFrameProxy> RemoteVideoFrameProxy::create(IPC::Connection& connection, RemoteVideoFrameObjectHeapProxy& videoFrameObjectHeapProxy, Properties&& properties)
{
    return adoptRef(*new RemoteVideoFrameProxy(connection, videoFrameObjectHeapProxy, WTFMove(properties)));
}

static void releaseRemoteVideoFrameProxy(IPC::Connection& connection, const RemoteVideoFrameWriteReference& reference)
{
    connection.send(Messages::RemoteVideoFrameObjectHeap::ReleaseVideoFrame(reference), 0, IPC::SendOption::DispatchMessageEvenWhenWaitingForSyncReply);
}

void RemoteVideoFrameProxy::releaseUnused(IPC::Connection& connection, Properties&& properties)
{
    releaseRemoteVideoFrameProxy(connection, { { properties.reference.identifier(), properties.reference.version() }, 0 });
}

RemoteVideoFrameProxy::RemoteVideoFrameProxy(IPC::Connection& connection, RemoteVideoFrameObjectHeapProxy& videoFrameObjectHeapProxy, Properties&& properties)
    : VideoFrame(properties.presentationTime, properties.isMirrored, properties.rotation, WTFMove(properties.colorSpace))
    , m_connection(&connection)
    , m_referenceTracker(properties.reference)
    , m_size(properties.size)
    , m_pixelFormat(properties.pixelFormat)
    , m_videoFrameObjectHeapProxy(&videoFrameObjectHeapProxy)
{
}

RemoteVideoFrameProxy::RemoteVideoFrameProxy(CloneConstructor, RemoteVideoFrameProxy& baseVideoFrame)
    : VideoFrame(baseVideoFrame.presentationTime(), baseVideoFrame.isMirrored(), baseVideoFrame.rotation(), WebCore::PlatformVideoColorSpace { baseVideoFrame.colorSpace() })
    , m_baseVideoFrame(&baseVideoFrame)
    , m_size(baseVideoFrame.m_size)
    , m_pixelFormat(baseVideoFrame.m_pixelFormat)
{
}

RemoteVideoFrameProxy::~RemoteVideoFrameProxy()
{
    if (m_connection)
        releaseRemoteVideoFrameProxy(*m_connection, m_referenceTracker->write());
}

RemoteVideoFrameIdentifier RemoteVideoFrameProxy::identifier() const
{
    return m_baseVideoFrame ? m_baseVideoFrame->m_referenceTracker->identifier() : m_referenceTracker->identifier();
}

RemoteVideoFrameReadReference RemoteVideoFrameProxy::newReadReference() const
{
    return m_baseVideoFrame ? m_baseVideoFrame->m_referenceTracker->read() : m_referenceTracker->read();
}

uint32_t RemoteVideoFrameProxy::pixelFormat() const
{
    return m_pixelFormat;
}

#if PLATFORM(COCOA)
CVPixelBufferRef RemoteVideoFrameProxy::pixelBuffer() const
{
    if (m_baseVideoFrame)
        return m_baseVideoFrame->pixelBuffer();

    Locker lock(m_pixelBufferLock);
    if (!m_pixelBuffer && m_videoFrameObjectHeapProxy) {
        auto videoFrameObjectHeapProxy = std::exchange(m_videoFrameObjectHeapProxy, nullptr);

        bool canSendSync = isMainRunLoop(); // FIXME: we should be able to sendSync from other threads too.
        bool canUseIOSurface = !WebProcess::singleton().shouldUseRemoteRenderingForWebGL();
        if (!canUseIOSurface || !canSendSync) {
            BinarySemaphore semaphore;
            videoFrameObjectHeapProxy->getVideoFrameBuffer(*this, canUseIOSurface, [this, &semaphore](auto pixelBuffer) {
                m_pixelBuffer = WTFMove(pixelBuffer);
                semaphore.signal();
            });
            semaphore.wait();
        } else {
            auto sendResult = m_connection->sendSync(Messages::RemoteVideoFrameObjectHeap::PixelBuffer(newReadReference()), 0, defaultTimeout);
            if (sendResult.succeeded())
                std::tie(m_pixelBuffer) = sendResult.takeReply();
        }
    }
    // FIXME: Some code paths do not like empty pixel buffers.
    if (!m_pixelBuffer)
        m_pixelBuffer = WebCore::createBlackPixelBuffer(static_cast<size_t>(m_size.width()), static_cast<size_t>(m_size.height()));
    return m_pixelBuffer.get();
}
#endif

Ref<WebCore::VideoFrame> RemoteVideoFrameProxy::clone()
{
    return adoptRef(*new RemoteVideoFrameProxy(cloneConstructor, *this));
}

TextStream& operator<<(TextStream& ts, const RemoteVideoFrameProxy::Properties& properties)
{
    ts << "{ reference=" << properties.reference
        << ", presentationTime=" << properties.presentationTime
        << ", isMirrored=" << properties.isMirrored
        << ", rotation=" << static_cast<int>(properties.rotation)
        << ", size=" << properties.size
        << ", pixelFormat=" << properties.pixelFormat
        << " }";
    return ts;
}

}
#endif
