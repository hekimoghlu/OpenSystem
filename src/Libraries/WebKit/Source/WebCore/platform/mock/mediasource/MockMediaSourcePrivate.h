/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 30, 2021.
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

#if ENABLE(MEDIA_SOURCE)

#include "MediaSourcePrivate.h"
#include <wtf/LoggerHelper.h>
#include <wtf/MediaTime.h>

namespace WebCore {

class MockMediaPlayerMediaSource;
class MockSourceBufferPrivate;

class MockMediaSourcePrivate final
    : public MediaSourcePrivate
#if !RELEASE_LOG_DISABLED
    , private LoggerHelper
#endif
{
public:
    static Ref<MockMediaSourcePrivate> create(MockMediaPlayerMediaSource&, MediaSourcePrivateClient&);
    virtual ~MockMediaSourcePrivate();

    constexpr MediaPlatformType platformType() const final { return MediaPlatformType::Mock; }

    RefPtr<MediaPlayerPrivateInterface> player() const final;
    void setPlayer(WebCore::MediaPlayerPrivateInterface*) final;

    std::optional<VideoPlaybackQualityMetrics> videoPlaybackQualityMetrics();

    void incrementTotalVideoFrames() { ++m_totalVideoFrames; }
    void incrementDroppedFrames() { ++m_droppedVideoFrames; }
    void incrementCorruptedFrames() { ++m_corruptedVideoFrames; }
    void incrementTotalFrameDelayBy(const MediaTime& delay) { m_totalFrameDelay += delay; }

#if !RELEASE_LOG_DISABLED
    const Logger& logger() const final { return m_logger.get(); }
    ASCIILiteral logClassName() const override { return "MockMediaSourcePrivate"_s; }
    uint64_t logIdentifier() const final { return m_logIdentifier; }
    WTFLogChannel& logChannel() const final;

    uint64_t nextSourceBufferLogIdentifier() { return childLogIdentifier(m_logIdentifier, ++m_nextSourceBufferID); }
#endif

private:
    MockMediaSourcePrivate(MockMediaPlayerMediaSource&, MediaSourcePrivateClient&);

    // MediaSourcePrivate Overrides
    AddStatus addSourceBuffer(const ContentType&, RefPtr<SourceBufferPrivate>&) override;
    void durationChanged(const MediaTime&) override;
    void markEndOfStream(EndOfStreamStatus) override;

    MediaPlayer::ReadyState mediaPlayerReadyState() const override;
    void setMediaPlayerReadyState(MediaPlayer::ReadyState) override;

    void notifyActiveSourceBuffersChanged() final;

    friend class MockSourceBufferPrivate;

    WeakPtr<MockMediaPlayerMediaSource> m_player;

    unsigned m_totalVideoFrames { 0 };
    unsigned m_droppedVideoFrames { 0 };
    unsigned m_corruptedVideoFrames { 0 };
    MediaTime m_totalFrameDelay;
#if !RELEASE_LOG_DISABLED
    Ref<const Logger> m_logger;
    const uint64_t m_logIdentifier;
    uint64_t m_nextSourceBufferID { 0 };
#endif
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::MockMediaSourcePrivate)
static bool isType(const WebCore::MediaSourcePrivate& mediaSource) { return mediaSource.platformType() == WebCore::MediaPlatformType::Mock; }
SPECIALIZE_TYPE_TRAITS_END()

#endif // ENABLE(MEDIA_SOURCE)
