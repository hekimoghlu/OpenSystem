/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 11, 2022.
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

#if USE(LIBWEBRTC)

#include "LibWebRTCMacros.h"
#include "MediaStreamTrackPrivate.h"
#include "Timer.h"
#include <wtf/Lock.h>

WTF_IGNORE_WARNINGS_IN_THIRD_PARTY_CODE_BEGIN

#include <webrtc/api/media_stream_interface.h>

WTF_IGNORE_WARNINGS_IN_THIRD_PARTY_CODE_END

#include <wtf/LoggerHelper.h>
#include <wtf/ThreadSafeRefCounted.h>

namespace WebCore {

class RealtimeOutgoingVideoSource
    : public ThreadSafeRefCounted<RealtimeOutgoingVideoSource, WTF::DestructionThread::Main>
    , public webrtc::VideoTrackSourceInterface
    , private MediaStreamTrackPrivateObserver
    , private RealtimeMediaSource::VideoFrameObserver
#if !RELEASE_LOG_DISABLED
    , private LoggerHelper
#endif
{
public:
    static Ref<RealtimeOutgoingVideoSource> create(Ref<MediaStreamTrackPrivate>&& videoSource);
    ~RealtimeOutgoingVideoSource();

    void start() { observeSource(); }
    void stop();
    void setSource(Ref<MediaStreamTrackPrivate>&&);
    MediaStreamTrackPrivate& source() const { return m_videoSource.get(); }

    void AddRef() const final { ref(); }
    webrtc::RefCountReleaseStatus Release() const final
    {
        deref();
        return webrtc::RefCountReleaseStatus::kOtherRefsRemained;
    }

    void applyRotation();
    void disableVideoScaling() { m_enableVideoFrameScaling = false; }

protected:
    explicit RealtimeOutgoingVideoSource(Ref<MediaStreamTrackPrivate>&&);

    void sendFrame(rtc::scoped_refptr<webrtc::VideoFrameBuffer>&&);
    bool isSilenced() const { return m_muted || !m_enabled; }

    virtual rtc::scoped_refptr<webrtc::VideoFrameBuffer> createBlackFrame(size_t width, size_t height) = 0;

    bool m_shouldApplyRotation { false };
    webrtc::VideoRotation m_currentRotation { webrtc::kVideoRotation_0 };

#if !RELEASE_LOG_DISABLED
    // LoggerHelper API
    const Logger& logger() const final { return m_logger.get(); }
    uint64_t logIdentifier() const final { return m_logIdentifier; }
    ASCIILiteral logClassName() const final { return "RealtimeOutgoingVideoSource"_s; }
    WTFLogChannel& logChannel() const final;
#endif

    double videoFrameScaling() const { return m_enableVideoFrameScaling ? (double)m_videoFrameScaling : 1; }

private:
    void sendBlackFramesIfNeeded();
    void sendOneBlackFrame();
    void initializeFromSource();
    void updateFramesSending();

    void observeSource();
    void unobserveSource();

    USING_CAN_MAKE_WEAKPTR(MediaStreamTrackPrivateObserver);

    // Notifier API
    void RegisterObserver(webrtc::ObserverInterface*) final { }
    void UnregisterObserver(webrtc::ObserverInterface*) final { }

    // VideoTrackSourceInterface API
    bool is_screencast() const final { return false; }
    std::optional<bool> needs_denoising() const final { return std::optional<bool>(); }
    bool GetStats(Stats*) final { return false; };
    bool SupportsEncodedOutput() const final { return false; }
    void GenerateKeyFrame() final { }
    void AddEncodedSink(rtc::VideoSinkInterface<webrtc::RecordableEncodedFrame>*) final { }
    void RemoveEncodedSink(rtc::VideoSinkInterface<webrtc::RecordableEncodedFrame>*) final { }

    // MediaSourceInterface API
    SourceState state() const final { return SourceState(); }
    bool remote() const final { return true; }

    // rtc::VideoSourceInterface<webrtc::VideoFrame> API
    void AddOrUpdateSink(rtc::VideoSinkInterface<webrtc::VideoFrame>*, const rtc::VideoSinkWants&) final;
    void RemoveSink(rtc::VideoSinkInterface<webrtc::VideoFrame>*) final;

    void sourceMutedChanged();
    void sourceEnabledChanged();
    void startObservingVideoFrames();

    // MediaStreamTrackPrivateObserver API
    void trackMutedChanged(MediaStreamTrackPrivate&) final { sourceMutedChanged(); }
    void trackEnabledChanged(MediaStreamTrackPrivate&) final { sourceEnabledChanged(); }
    void trackSettingsChanged(MediaStreamTrackPrivate&) final { initializeFromSource(); }
    void trackEnded(MediaStreamTrackPrivate&) final { }

    // RealtimeMediaSource::VideoFrameObserver API
    void videoFrameAvailable(VideoFrame&, VideoFrameTimeMetadata) override { }

    Ref<MediaStreamTrackPrivate> m_videoSource;
    Timer m_blackFrameTimer;
    rtc::scoped_refptr<webrtc::VideoFrameBuffer> m_blackFrame;

    mutable Lock m_sinksLock;
    UncheckedKeyHashSet<rtc::VideoSinkInterface<webrtc::VideoFrame>*> m_sinks WTF_GUARDED_BY_LOCK(m_sinksLock);
    bool m_areSinksAskingToApplyRotation { false };

    bool m_enabled { true };
    bool m_muted { false };
    uint32_t m_width { 0 };
    uint32_t m_height { 0 };
    std::optional<double> m_maxFrameRate;
    std::optional<double> m_maxPixelCount;
    std::atomic<double> m_videoFrameScaling { 1.0 };
    bool m_enableVideoFrameScaling { true };
    bool m_isObservingVideoFrames { false };

#if !RELEASE_LOG_DISABLED
    Ref<const Logger> m_logger;
    const uint64_t m_logIdentifier;
    MonotonicTime m_lastFrameLogTime;
    unsigned m_frameCount { 0 };
#endif
};

} // namespace WebCore

#endif // USE(LIBWEBRTC)
