/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 22, 2024.
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

#if ENABLE(MEDIA_STREAM) && USE(AVFOUNDATION)

#include "FrameRateMonitor.h"
#include "SampleBufferDisplayLayer.h"
#include "VideoFrame.h"
#include <wtf/Deque.h>
#include <wtf/Forward.h>
#include <wtf/RetainPtr.h>
#include <wtf/MediaTime.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/WorkQueue.h>

OBJC_CLASS AVSampleBufferDisplayLayer;
OBJC_CLASS WebAVSampleBufferStatusChangeListener;

typedef struct __CVBuffer* CVPixelBufferRef;

namespace WebCore {

enum class VideoFrameRotation : uint16_t;

class WEBCORE_EXPORT LocalSampleBufferDisplayLayer final : public SampleBufferDisplayLayer {
    WTF_MAKE_TZONE_ALLOCATED_EXPORT(LocalSampleBufferDisplayLayer, WEBCORE_EXPORT);
public:
    static RefPtr<LocalSampleBufferDisplayLayer> create(SampleBufferDisplayLayerClient&);

    LocalSampleBufferDisplayLayer(RetainPtr<AVSampleBufferDisplayLayer>&&, SampleBufferDisplayLayerClient&);
    ~LocalSampleBufferDisplayLayer();

    // API used by WebAVSampleBufferStatusChangeListener
    void layerStatusDidChange();
    void layerErrorDidChange();
    void rootLayerBoundsDidChange();

    CGRect bounds() const;

    PlatformLayer* displayLayer();

    void updateSampleLayerBoundsAndPosition(std::optional<CGRect>);

    // SampleBufferDisplayLayer.
    PlatformLayer* rootLayer() final;
    void initialize(bool hideRootLayer, IntSize, bool shouldMaintainAspectRatio, CompletionHandler<void(bool didSucceed)>&&) final;
#if !RELEASE_LOG_DISABLED
    void setLogIdentifier(String&& logIdentifier) final { m_logIdentifier = WTFMove(logIdentifier); }
#endif
    bool didFail() const final;

    void updateDisplayMode(bool hideDisplayLayer, bool hideRootLayer) final;
    void updateBoundsAndPosition(CGRect, std::optional<WTF::MachSendRight>&&) final;

    void flush() final;
    void flushAndRemoveImage() final;

    void play() final;
    void pause() final;

    void enqueueVideoFrame(VideoFrame&) final;
    void clearVideoFrames() final;
    void setRenderPolicy(RenderPolicy) final;
    void setShouldMaintainAspectRatio(bool) final;

private:
    void enqueueBufferInternal(CVPixelBufferRef, MediaTime);
    void removeOldVideoFramesFromPendingQueue();
    void addVideoFrameToPendingQueue(Ref<VideoFrame>&&);
    void requestNotificationWhenReadyForVideoData();

#if !RELEASE_LOG_DISABLED
    void onIrregularFrameRateNotification(MonotonicTime frameTime, MonotonicTime lastFrameTime);
#endif

    WorkQueue& workQueue() { return m_processingQueue.get(); }

private:
    RetainPtr<WebAVSampleBufferStatusChangeListener> m_statusChangeListener;
    RetainPtr<AVSampleBufferDisplayLayer> m_sampleBufferDisplayLayer;
    RetainPtr<PlatformLayer> m_rootLayer;
    RenderPolicy m_renderPolicy { RenderPolicy::TimingInfo };

    Ref<WorkQueue> m_processingQueue;

    // Only accessed through m_processingQueue or if m_processingQueue is null.
    using PendingSampleQueue = Deque<Ref<VideoFrame>>;
    PendingSampleQueue m_pendingVideoFrameQueue;

    WebCore::VideoFrameRotation m_videoFrameRotation { WebCore::VideoFrameRotation::None };
    CGAffineTransform m_affineTransform { CGAffineTransformIdentity };

    bool m_paused { false };
    bool m_didFail { false };
#if !RELEASE_LOG_DISABLED
    String m_logIdentifier;
    FrameRateMonitor m_frameRateMonitor WTF_GUARDED_BY_CAPABILITY(workQueue());
#endif
};

inline void LocalSampleBufferDisplayLayer::setRenderPolicy(RenderPolicy renderPolicy)
{
    m_renderPolicy = renderPolicy;
}

inline void LocalSampleBufferDisplayLayer::play()
{
    m_paused = false;
}

inline void LocalSampleBufferDisplayLayer::pause()
{
    m_paused = true;
}


}

#endif // ENABLE(MEDIA_STREAM) && USE(AVFOUNDATION)
