/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 25, 2024.
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

#include "ExceptionOr.h"
#include "MediaRecorderPrivateOptions.h"
#include "RealtimeMediaSource.h"
#include <wtf/CheckedRef.h>
#include <wtf/CompletionHandler.h>
#include <wtf/Forward.h>
#include <wtf/TZoneMalloc.h>

#if ENABLE(MEDIA_RECORDER)

namespace WTF {
class MediaTime;
}

namespace WebCore {

class AudioStreamDescription;
class MediaSample;
class MediaStreamPrivate;
class MediaStreamTrackPrivate;
class PlatformAudioData;
class FragmentedSharedBuffer;

struct MediaRecorderPrivateOptions;

class MediaRecorderPrivate
    : public RealtimeMediaSource::AudioSampleObserver
    , public RealtimeMediaSource::VideoFrameObserver
    , public CanMakeCheckedPtr<MediaRecorderPrivate> {
    WTF_MAKE_TZONE_ALLOCATED(MediaRecorderPrivate);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(MediaRecorderPrivate);
public:
    ~MediaRecorderPrivate();

    // CheckedPtr interface
    uint32_t checkedPtrCount() const final { return CanMakeCheckedPtr::checkedPtrCount(); }
    uint32_t checkedPtrCountWithoutThreadCheck() const final { return CanMakeCheckedPtr::checkedPtrCountWithoutThreadCheck(); }
    void incrementCheckedPtrCount() const final { CanMakeCheckedPtr::incrementCheckedPtrCount(); }
    void decrementCheckedPtrCount() const final { CanMakeCheckedPtr::decrementCheckedPtrCount(); }

    struct AudioVideoSelectedTracks {
        MediaStreamTrackPrivate* audioTrack { nullptr };
        MediaStreamTrackPrivate* videoTrack { nullptr };
    };
    WEBCORE_EXPORT static AudioVideoSelectedTracks selectTracks(MediaStreamPrivate&);

    using FetchDataCallback = CompletionHandler<void(RefPtr<FragmentedSharedBuffer>&&, const String& mimeType, double)>;
    virtual void fetchData(FetchDataCallback&&) = 0;
    virtual String mimeType() const = 0;

    void stop(CompletionHandler<void()>&&);
    void pause(CompletionHandler<void()>&&);
    void resume(CompletionHandler<void()>&&);

    using StartRecordingCallback = CompletionHandler<void(ExceptionOr<String>&&, unsigned, unsigned)>;
    virtual void startRecording(StartRecordingCallback&& callback) { callback(String(mimeType()), 0, 0); }

    void trackMutedChanged(MediaStreamTrackPrivate& track) { checkTrackState(track); }
    void trackEnabledChanged(MediaStreamTrackPrivate& track) { checkTrackState(track); }

    struct BitRates {
        unsigned audio;
        unsigned video;
    };
    static BitRates computeBitRates(const MediaRecorderPrivateOptions&, const MediaStreamPrivate* = nullptr);

protected:
    void setAudioSource(RefPtr<RealtimeMediaSource>&&);
    void setVideoSource(RefPtr<RealtimeMediaSource>&&);

    void checkTrackState(const MediaStreamTrackPrivate&);

    bool shouldMuteAudio() const { return m_shouldMuteAudio; }
    bool shouldMuteVideo() const { return m_shouldMuteVideo; }

private:

    virtual void stopRecording(CompletionHandler<void()>&&) = 0;
    virtual void pauseRecording(CompletionHandler<void()>&&) = 0;
    virtual void resumeRecording(CompletionHandler<void()>&&) = 0;

private:
    bool m_shouldMuteAudio { false };
    bool m_shouldMuteVideo { false };
    RefPtr<RealtimeMediaSource> m_audioSource;
    RefPtr<RealtimeMediaSource> m_videoSource;
    RefPtr<RealtimeMediaSource> m_pausedAudioSource;
    RefPtr<RealtimeMediaSource> m_pausedVideoSource;
};

inline void MediaRecorderPrivate::setAudioSource(RefPtr<RealtimeMediaSource>&& audioSource)
{
    if (m_audioSource)
        m_audioSource->removeAudioSampleObserver(*this);

    m_audioSource = WTFMove(audioSource);

    if (m_audioSource)
        m_audioSource->addAudioSampleObserver(*this);
}

inline void MediaRecorderPrivate::setVideoSource(RefPtr<RealtimeMediaSource>&& videoSource)
{
    if (m_videoSource)
        m_videoSource->removeVideoFrameObserver(*this);

    m_videoSource = WTFMove(videoSource);

    if (m_videoSource)
        m_videoSource->addVideoFrameObserver(*this);
}

inline MediaRecorderPrivate::~MediaRecorderPrivate()
{
    // Subclasses should stop observing sonner than here. Otherwise they might be called from a background thread while half destroyed
    ASSERT(!m_audioSource);
    ASSERT(!m_videoSource);
    if (m_audioSource)
        m_audioSource->removeAudioSampleObserver(*this);
    if (m_videoSource)
        m_videoSource->removeVideoFrameObserver(*this);
}

} // namespace WebCore

#endif // ENABLE(MEDIA_RECORDER)
