/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 17, 2024.
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
#include "MediaRecorderPrivate.h"

#if ENABLE(MEDIA_RECORDER)

#include "MediaStreamPrivate.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(MediaRecorderPrivate);

constexpr unsigned SmallAudioBitRate = 8000;
constexpr unsigned SmallVideoBitRate = 80000;
constexpr unsigned LargeAudioBitRate = 192000;
constexpr unsigned LargeVideoBitRate = 10000000;

MediaRecorderPrivate::AudioVideoSelectedTracks MediaRecorderPrivate::selectTracks(MediaStreamPrivate& stream)
{
    AudioVideoSelectedTracks selectedTracks;
    stream.forEachTrack([&](auto& track) {
        if (track.ended())
            return;
        switch (track.type()) {
        case RealtimeMediaSource::Type::Video: {
            auto& settings = track.settings();
            if (!selectedTracks.videoTrack && settings.supportsWidth() && settings.supportsHeight())
                selectedTracks.videoTrack = &track;
            break;
        }
        case RealtimeMediaSource::Type::Audio:
            if (!selectedTracks.audioTrack)
                selectedTracks.audioTrack = &track;
            break;
        }
    });
    return selectedTracks;
}

void MediaRecorderPrivate::checkTrackState(const MediaStreamTrackPrivate& track)
{
    if (track.hasSource(m_audioSource.get())) {
        m_shouldMuteAudio = track.muted() || !track.enabled();
        return;
    }
    if (track.hasSource(m_videoSource.get()))
        m_shouldMuteVideo = track.muted() || !track.enabled();
}

void MediaRecorderPrivate::stop(CompletionHandler<void()>&& completionHandler)
{
    setAudioSource(nullptr);
    setVideoSource(nullptr);
    stopRecording(WTFMove(completionHandler));
}

void MediaRecorderPrivate::pause(CompletionHandler<void()>&& completionHandler)
{
    ASSERT(!m_pausedAudioSource);
    ASSERT(!m_pausedVideoSource);

    m_pausedAudioSource = m_audioSource;
    m_pausedVideoSource = m_videoSource;

    setAudioSource(nullptr);
    setVideoSource(nullptr);

    pauseRecording(WTFMove(completionHandler));
}

void MediaRecorderPrivate::resume(CompletionHandler<void()>&& completionHandler)
{
    ASSERT(m_pausedAudioSource || m_pausedVideoSource);

    setAudioSource(WTFMove(m_pausedAudioSource));
    setVideoSource(WTFMove(m_pausedVideoSource));

    resumeRecording(WTFMove(completionHandler));
}

MediaRecorderPrivate::BitRates MediaRecorderPrivate::computeBitRates(const MediaRecorderPrivateOptions& options, const MediaStreamPrivate* stream)
{
    if (options.bitsPerSecond) {
        bool hasAudio = stream ? stream->hasAudio() : true;
        bool hasVideo = stream ? stream->hasVideo() : true;
        auto totalBitsPerSecond = *options.bitsPerSecond;

        if (hasAudio && hasVideo) {
            auto audioBitsPerSecond =  std::min(LargeAudioBitRate, std::max(SmallAudioBitRate, totalBitsPerSecond / 10));
            auto remainingBitsPerSecond = totalBitsPerSecond > audioBitsPerSecond ? (totalBitsPerSecond - audioBitsPerSecond) : 0;
            return { audioBitsPerSecond, std::max(remainingBitsPerSecond, SmallVideoBitRate) };
        }

        if (hasAudio)
            return { std::max(SmallAudioBitRate, totalBitsPerSecond), 0 };

        return { 0, std::max(SmallVideoBitRate, totalBitsPerSecond) };
    }

    return { options.audioBitsPerSecond.value_or(LargeAudioBitRate), options.videoBitsPerSecond.value_or(LargeVideoBitRate) };
}

} // namespace WebCore

#endif // ENABLE(MEDIA_RECORDER)
