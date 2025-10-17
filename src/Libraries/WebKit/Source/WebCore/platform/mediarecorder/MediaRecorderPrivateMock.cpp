/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 25, 2024.
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
#include "MediaRecorderPrivateMock.h"

#if ENABLE(MEDIA_RECORDER)

#include "MediaStreamTrackPrivate.h"
#include "SharedBuffer.h"
#include "Timer.h"
#include <wtf/MediaTime.h>
#include <wtf/MonotonicTime.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(MediaRecorderPrivateMock);

MediaRecorderPrivateMock::MediaRecorderPrivateMock(MediaStreamPrivate& stream)
{
    auto selectedTracks = MediaRecorderPrivate::selectTracks(stream);
    if (selectedTracks.audioTrack) {
        m_audioTrackID = selectedTracks.audioTrack->id();
        setAudioSource(&selectedTracks.audioTrack->source());
    }
    if (selectedTracks.videoTrack) {
        m_videoTrackID = selectedTracks.videoTrack->id();
        setVideoSource(&selectedTracks.videoTrack->source());
    }
}

MediaRecorderPrivateMock::~MediaRecorderPrivateMock()
{
}

void MediaRecorderPrivateMock::stopRecording(CompletionHandler<void()>&& completionHandler)
{
    completionHandler();
}

void MediaRecorderPrivateMock::pauseRecording(CompletionHandler<void()>&& completionHandler)
{
    completionHandler();
}

void MediaRecorderPrivateMock::resumeRecording(CompletionHandler<void()>&& completionHandler)
{
    completionHandler();
}

void MediaRecorderPrivateMock::videoFrameAvailable(VideoFrame&, VideoFrameTimeMetadata)
{
    Locker locker { m_bufferLock };
    m_buffer.append("Video Track ID: "_s, m_videoTrackID);
    generateMockCounterString();
}

void MediaRecorderPrivateMock::audioSamplesAvailable(const MediaTime&, const PlatformAudioData&, const AudioStreamDescription&, size_t)
{
    // Heap allocations are forbidden on the audio thread for performance reasons so we need to
    // explicitly allow the following allocation(s).
    DisableMallocRestrictionsForCurrentThreadScope disableMallocRestrictions;
    Locker locker { m_bufferLock };
    m_buffer.append("Audio Track ID: "_s, m_audioTrackID);
    generateMockCounterString();
}

void MediaRecorderPrivateMock::generateMockCounterString()
{
    m_buffer.append(" Counter: "_s, ++m_counter, "\r\n---------\r\n"_s);
}

void MediaRecorderPrivateMock::fetchData(FetchDataCallback&& completionHandler)
{
    RefPtr<FragmentedSharedBuffer> buffer;
    {
        Locker locker { m_bufferLock };
        Vector<uint8_t> value(m_buffer.span<uint8_t>());
        m_buffer.clear();
        buffer = SharedBuffer::create(WTFMove(value));
    }

    // Delay calling the completion handler a bit to mimick real writer behavior.
    Timer::schedule(50_ms, [completionHandler = WTFMove(completionHandler), buffer = WTFMove(buffer), mimeType = mimeType(), timeCode = MonotonicTime::now().secondsSinceEpoch().value()]() mutable {
        completionHandler(WTFMove(buffer), mimeType, timeCode);
    });
}

String MediaRecorderPrivateMock::mimeType() const
{
    static NeverDestroyed<const String> textPlainMimeType(MAKE_STATIC_STRING_IMPL("text/plain"));
    return textPlainMimeType;
}

} // namespace WebCore

#endif // ENABLE(MEDIA_RECORDER)
