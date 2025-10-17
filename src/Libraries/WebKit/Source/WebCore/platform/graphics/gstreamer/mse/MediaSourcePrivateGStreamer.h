/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 11, 2024.
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

#if ENABLE(MEDIA_SOURCE) && USE(GSTREAMER)
#include "MediaSourcePrivate.h"

#include <wtf/Forward.h>
#include <wtf/HashSet.h>
#include <wtf/LoggerHelper.h>

namespace WebCore {

class SourceBufferPrivateGStreamer;
class MediaPlayerPrivateGStreamerMSE;
class PlatformTimeRanges;

class MediaSourcePrivateGStreamer final : public MediaSourcePrivate
#if !RELEASE_LOG_DISABLED
    , private LoggerHelper
#endif
{
public:
    static Ref<MediaSourcePrivateGStreamer> open(MediaSourcePrivateClient&, MediaPlayerPrivateGStreamerMSE&);
    virtual ~MediaSourcePrivateGStreamer();

    void setPlayer(MediaPlayerPrivateInterface*) final;
    RefPtr<MediaPlayerPrivateInterface> player() const final;

    constexpr MediaPlatformType platformType() const final { return MediaPlatformType::GStreamer; }

    AddStatus addSourceBuffer(const ContentType&, RefPtr<SourceBufferPrivate>&) override;

    void durationChanged(const MediaTime&) override;
    void markEndOfStream(EndOfStreamStatus) override;
    void unmarkEndOfStream() override;

    MediaPlayer::ReadyState mediaPlayerReadyState() const override;
    void setMediaPlayerReadyState(MediaPlayer::ReadyState) override;

    void notifyActiveSourceBuffersChanged() final;

    void startPlaybackIfHasAllTracks();
    bool hasAllTracks() const { return m_hasAllTracks; }

    void detach();

    TrackID registerTrackId(TrackID);
    bool unregisterTrackId(TrackID);

#if !RELEASE_LOG_DISABLED
    const Logger& logger() const final { return m_logger; }
    ASCIILiteral logClassName() const override { return "MediaSourcePrivateGStreamer"_s; }
    uint64_t logIdentifier() const final { return m_logIdentifier; }
    WTFLogChannel& logChannel() const final;

    uint64_t nextSourceBufferLogIdentifier() { return childLogIdentifier(m_logIdentifier, ++m_nextSourceBufferID); }
#endif

private:
    MediaSourcePrivateGStreamer(MediaSourcePrivateClient&, MediaPlayerPrivateGStreamerMSE&);
    RefPtr<MediaPlayerPrivateGStreamerMSE> platformPlayer() const;

    ThreadSafeWeakPtr<MediaPlayerPrivateGStreamerMSE> m_playerPrivate;
    bool m_hasAllTracks { false };
#if !RELEASE_LOG_DISABLED
    Ref<const Logger> m_logger;
    const uint64_t m_logIdentifier;
#endif

    uint64_t m_nextSourceBufferID { 0 };

    // Stores known track IDs, so we can work around ID collisions between multiple source buffers.
    // The registry is placed here to enforce ID uniqueness specifically by player, not by process,
    // since its not an issue if multiple players use the same ID, and we want to preserve IDs as much as possible.
    UncheckedKeyHashSet<TrackID, WTF::IntHash<TrackID>, WTF::UnsignedWithZeroKeyHashTraits<TrackID>> m_trackIdRegistry;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::MediaSourcePrivateGStreamer)
static bool isType(const WebCore::MediaSourcePrivate& mediaSource) { return mediaSource.platformType() == WebCore::MediaPlatformType::GStreamer; }
SPECIALIZE_TYPE_TRAITS_END()

#endif
