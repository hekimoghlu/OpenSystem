/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 17, 2022.
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

#include "ProcessIdentity.h"
#include "SampleMap.h"
#include <CoreMedia/CMTime.h>
#include <wtf/Function.h>
#include <wtf/Lock.h>
#include <wtf/MonotonicTime.h>
#include <wtf/Ref.h>
#include <wtf/RetainPtr.h>
#include <wtf/ThreadSafeWeakPtr.h>

OBJC_CLASS AVSampleBufferDisplayLayer;
OBJC_CLASS AVSampleBufferVideoRenderer;
OBJC_PROTOCOL(WebSampleBufferVideoRendering);
typedef struct opaqueCMBufferQueue *CMBufferQueueRef;
typedef struct opaqueCMSampleBuffer *CMSampleBufferRef;
typedef struct OpaqueCMTimebase* CMTimebaseRef;
typedef struct __CVBuffer* CVPixelBufferRef;

namespace WTF {
class WorkQueue;
}

namespace WebCore {

class EffectiveRateChangedListener;
class MediaSample;
class WebCoreDecompressionSession;

class VideoMediaSampleRenderer : public ThreadSafeRefCountedAndCanMakeThreadSafeWeakPtr<VideoMediaSampleRenderer, WTF::DestructionThread::Main> {
public:
    static Ref<VideoMediaSampleRenderer> create(WebSampleBufferVideoRendering *renderer) { return adoptRef(*new VideoMediaSampleRenderer(renderer)); }
    ~VideoMediaSampleRenderer();

    bool prefersDecompressionSession() const { return m_prefersDecompressionSession; }
    void setPrefersDecompressionSession(bool);
    bool isUsingDecompressionSession() const;

    void setTimebase(RetainPtr<CMTimebaseRef>&&);
    RetainPtr<CMTimebaseRef> timebase() const;

    bool isReadyForMoreMediaData() const;
    void requestMediaDataWhenReady(Function<void()>&&);
    void enqueueSample(const MediaSample&);
    void stopRequestingMediaData();

    void notifyWhenHasAvailableVideoFrame(Function<void(const MediaTime&, double)>&&);
    void notifyWhenDecodingErrorOccurred(Function<void(OSStatus)>&&);

    void flush();

    void expectMinimumUpcomingSampleBufferPresentationTime(const MediaTime&);
    void resetUpcomingSampleBufferPresentationTimeExpectations();

    WebSampleBufferVideoRendering *renderer() const;

    template <typename T> T* as() const;
    template <> AVSampleBufferVideoRenderer* as() const;
    template <> AVSampleBufferDisplayLayer* as() const { return m_displayLayer.get(); }

    struct DisplayedPixelBufferEntry {
        RetainPtr<CVPixelBufferRef> pixelBuffer;
        MediaTime presentationTimeStamp;
    };
    DisplayedPixelBufferEntry copyDisplayedPixelBuffer();

    unsigned totalDisplayedFrames() const;
    unsigned totalVideoFrames() const;
    unsigned droppedVideoFrames() const;
    unsigned corruptedVideoFrames() const;
    MediaTime totalFrameDelay() const;

    void setResourceOwner(const ProcessIdentity&);

private:
    VideoMediaSampleRenderer(WebSampleBufferVideoRendering *);

    void clearTimebase();
    using TimebaseAndTimerSource = std::pair<RetainPtr<CMTimebaseRef>, OSObjectPtr<dispatch_source_t>>;
    TimebaseAndTimerSource timebaseAndTimerSource() const;

    WebSampleBufferVideoRendering *rendererOrDisplayLayer() const;

    void resetReadyForMoreSample();
    void initializeDecompressionSession();
    void decodeNextSample();
    using FlushId = int;
    void decodedFrameAvailable(RetainPtr<CMSampleBufferRef>&&, FlushId);
    enum class DecodedFrameResult : uint8_t {
        TooEarly,
        TooLate,
        AlreadyDisplayed,
        Displayed
    };
    DecodedFrameResult maybeQueueFrameForDisplay(const CMTime&, CMSampleBufferRef, FlushId);
    void flushCompressedSampleQueue();
    void flushDecodedSampleQueue();
    void cancelTimer();
    void purgeDecodedSampleQueueAndDisplay(FlushId);
    bool purgeDecodedSampleQueue(const CMTime&);
    void schedulePurgeAndDisplayAtTime(const CMTime&);
    void maybeReschedulePurgeAndDisplay(FlushId);
    void enqueueDecodedSample(RetainPtr<CMSampleBufferRef>&&);
    size_t decodedSamplesCount() const;
    RetainPtr<CMSampleBufferRef> nextDecodedSample() const;
    CMTime nextDecodedSampleEndTime() const;

    void assignResourceOwner(CMSampleBufferRef);
    bool areSamplesQueuesReadyForMoreMediaData(size_t waterMark) const;
    void maybeBecomeReadyForMoreMediaData();
    bool shouldDecodeSample(CMSampleBufferRef, bool displaying);

    void notifyHasAvailableVideoFrame(const MediaTime&, double, FlushId);
    void notifyErrorHasOccurred(OSStatus);

    Ref<GuaranteedSerialFunctionDispatcher> dispatcher() const;
    void ensureOnDispatcher(Function<void()>&&) const;
    void ensureOnDispatcherSync(Function<void()>&&) const;
    dispatch_queue_t dispatchQueue() const;

    const RefPtr<WTF::WorkQueue> m_workQueue;
    RetainPtr<AVSampleBufferDisplayLayer> m_displayLayer;
#if HAVE(AVSAMPLEBUFFERVIDEORENDERER)
    RetainPtr<AVSampleBufferVideoRenderer> m_renderer;
#endif
    mutable Lock m_lock;
    TimebaseAndTimerSource m_timebaseAndTimerSource WTF_GUARDED_BY_LOCK(m_lock);
    RefPtr<EffectiveRateChangedListener> m_effectiveRateChangedListener;
    std::atomic<ssize_t> m_framesBeingDecoded { 0 };
    std::atomic<FlushId> m_flushId { 0 };
    Deque<std::pair<RetainPtr<CMSampleBufferRef>, FlushId>> m_compressedSampleQueue WTF_GUARDED_BY_CAPABILITY(dispatcher().get());
    RetainPtr<CMBufferQueueRef> m_decodedSampleQueue; // created on the main thread, immutable after creation.
    RefPtr<WebCoreDecompressionSession> m_decompressionSession;
    bool m_isDecodingSample WTF_GUARDED_BY_CAPABILITY(dispatcher().get()) { false };
    bool m_isDisplayingSample WTF_GUARDED_BY_CAPABILITY(dispatcher().get()) { false };
    bool m_forceLateSampleToBeDisplayed WTF_GUARDED_BY_CAPABILITY(dispatcher().get()) { false };
    std::optional<CMTime> m_lastDisplayedTime WTF_GUARDED_BY_CAPABILITY(dispatcher().get());
    std::optional<CMTime> m_lastDisplayedSample WTF_GUARDED_BY_CAPABILITY(dispatcher().get());
    std::optional<CMTime> m_nextScheduledPurge WTF_GUARDED_BY_CAPABILITY(dispatcher().get());

    Function<void()> m_readyForMoreSampleFunction;
    bool m_prefersDecompressionSession { false };
    std::optional<uint32_t> m_currentCodec;
    std::atomic<bool> m_gotDecodingError { false };

    // Playback Statistics
    std::atomic<unsigned> m_totalVideoFrames { 0 };
    std::atomic<unsigned> m_droppedVideoFrames { 0 };
    std::atomic<unsigned> m_corruptedVideoFrames { 0 };
    std::atomic<unsigned> m_presentedVideoFrames { 0 };
    MediaTime m_totalFrameDelay { MediaTime::zeroTime() };

    Function<void(const MediaTime&, double)> m_hasAvailableFrameCallback WTF_GUARDED_BY_CAPABILITY(mainThread);
    Function<void(OSStatus)> m_errorOccurredFunction WTF_GUARDED_BY_CAPABILITY(mainThread);
    ProcessIdentity m_resourceOwner;
    MonotonicTime m_startupTime;
};

} // namespace WebCore
