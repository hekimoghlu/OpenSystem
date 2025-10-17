/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 22, 2024.
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
#include "SampleBufferDisplayLayer.h"

#if PLATFORM(COCOA) && ENABLE(GPU_PROCESS) && ENABLE(MEDIA_STREAM)

#include "GPUConnectionToWebProcessMessages.h"
#include "GPUProcessConnection.h"
#include "LayerHostingContext.h"
#include "Logging.h"
#include "RemoteSampleBufferDisplayLayerManagerMessages.h"
#include "RemoteSampleBufferDisplayLayerMessages.h"
#include "RemoteVideoFrameProxy.h"
#include "SampleBufferDisplayLayerManager.h"
#include "WebProcess.h"

namespace WebKit {
using namespace WebCore;

Ref<SampleBufferDisplayLayer> SampleBufferDisplayLayer::create(SampleBufferDisplayLayerManager& manager, WebCore::SampleBufferDisplayLayerClient& client)
{
    return adoptRef(*new SampleBufferDisplayLayer(manager, client));
}

SampleBufferDisplayLayer::SampleBufferDisplayLayer(SampleBufferDisplayLayerManager& manager, WebCore::SampleBufferDisplayLayerClient& client)
    : WebCore::SampleBufferDisplayLayer(client)
    , m_gpuProcessConnection(&WebProcess::singleton().ensureGPUProcessConnection())
    , m_manager(manager)
    , m_connection(m_gpuProcessConnection.get()->connection())
{
    manager.addLayer(*this);
    auto gpuProcessConnection = m_gpuProcessConnection.get();
    gpuProcessConnection->addClient(*this);
}

void SampleBufferDisplayLayer::initialize(bool hideRootLayer, IntSize size, bool shouldMaintainAspectRatio, CompletionHandler<void(bool)>&& callback)
{
    m_connection->sendWithAsyncReply(Messages::RemoteSampleBufferDisplayLayerManager::CreateLayer { identifier(), hideRootLayer, size, shouldMaintainAspectRatio, canShowWhileLocked() }, [this, weakThis = WeakPtr { *this }, callback = WTFMove(callback)](auto contextId) mutable {
        if (!weakThis)
            return callback(false);
        m_hostingContextID = contextId;
        callback(!!contextId);
    });
}

#if !RELEASE_LOG_DISABLED
void SampleBufferDisplayLayer::setLogIdentifier(String&& logIdentifier)
{
    ASSERT(m_hostingContextID);
    m_connection->send(Messages::RemoteSampleBufferDisplayLayer::SetLogIdentifier { logIdentifier }, identifier());
}
#endif

SampleBufferDisplayLayer::~SampleBufferDisplayLayer()
{
    m_connection->send(Messages::RemoteSampleBufferDisplayLayerManager::ReleaseLayer { identifier() }, 0);
    if (RefPtr manager = m_manager.get())
        manager->removeLayer(*this);
}

bool SampleBufferDisplayLayer::didFail() const
{
    return m_didFail;
}

void SampleBufferDisplayLayer::updateDisplayMode(bool hideDisplayLayer, bool hideRootLayer)
{
    m_connection->send(Messages::RemoteSampleBufferDisplayLayer::UpdateDisplayMode { hideDisplayLayer, hideRootLayer }, identifier());
}

void SampleBufferDisplayLayer::updateBoundsAndPosition(CGRect bounds, std::optional<WTF::MachSendRight>&& fence)
{
    m_connection->send(Messages::GPUConnectionToWebProcess::UpdateSampleBufferDisplayLayerBoundsAndPosition { identifier(), bounds, WTFMove(fence) }, identifier());
}

void SampleBufferDisplayLayer::flush()
{
    m_connection->send(Messages::RemoteSampleBufferDisplayLayer::Flush { }, identifier());
}

void SampleBufferDisplayLayer::flushAndRemoveImage()
{
    m_connection->send(Messages::RemoteSampleBufferDisplayLayer::FlushAndRemoveImage { }, identifier());
}

void SampleBufferDisplayLayer::play()
{
    m_paused = false;
    m_connection->send(Messages::RemoteSampleBufferDisplayLayer::Play { }, identifier());
}

void SampleBufferDisplayLayer::pause()
{
    m_paused = true;
    m_connection->send(Messages::RemoteSampleBufferDisplayLayer::Pause { }, identifier());
}

void SampleBufferDisplayLayer::enqueueBlackFrameFrom(const VideoFrame& videoFrame)
{
    auto size = videoFrame.presentationSize();
    WebCore::IntSize blackFrameSize { static_cast<int>(size.width()), static_cast<int>(size.height()) };
    SharedVideoFrame sharedVideoFrame { videoFrame.presentationTime(), false, videoFrame.rotation(), blackFrameSize };
    m_connection->send(Messages::RemoteSampleBufferDisplayLayer::EnqueueVideoFrame { WTFMove(sharedVideoFrame) }, identifier());
}

void SampleBufferDisplayLayer::enqueueVideoFrame(VideoFrame& videoFrame)
{
    if (m_paused)
        return;

    auto sharedVideoFrame = m_sharedVideoFrameWriter.write(videoFrame,
        [this](auto& semaphore) { m_connection->send(Messages::RemoteSampleBufferDisplayLayer::SetSharedVideoFrameSemaphore { semaphore }, identifier()); },
        [this](SharedMemory::Handle&& handle) { m_connection->send(Messages::RemoteSampleBufferDisplayLayer::SetSharedVideoFrameMemory { WTFMove(handle) }, identifier()); }
    );
    if (!sharedVideoFrame)
        return;

    m_connection->send(Messages::RemoteSampleBufferDisplayLayer::EnqueueVideoFrame { WTFMove(*sharedVideoFrame) }, identifier());
}

void SampleBufferDisplayLayer::clearVideoFrames()
{
    m_connection->send(Messages::RemoteSampleBufferDisplayLayer::ClearVideoFrames { }, identifier());
}

PlatformLayer* SampleBufferDisplayLayer::rootLayer()
{
    if (!m_videoLayer && m_hostingContextID)
        m_videoLayer = LayerHostingContext::createPlatformLayerForHostingContext(*m_hostingContextID);
    return m_videoLayer.get();
}

void SampleBufferDisplayLayer::setDidFail(bool value)
{
    m_didFail = value;
    RefPtr client = m_client.get();
    if (client && m_didFail)
        client->sampleBufferDisplayLayerStatusDidFail();
}

void SampleBufferDisplayLayer::gpuProcessConnectionDidClose(GPUProcessConnection&)
{
    m_sharedVideoFrameWriter.disable();
    setDidFail(true);
}

void SampleBufferDisplayLayer::setShouldMaintainAspectRatio(bool shouldMaintainAspectRatio)
{
    m_connection->send(Messages::RemoteSampleBufferDisplayLayer::SetShouldMaintainAspectRatio { shouldMaintainAspectRatio }, identifier());
}

}

#endif
