/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 4, 2023.
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
#ifndef MockMediaPlayerMediaSource_h
#define MockMediaPlayerMediaSource_h

#if ENABLE(MEDIA_SOURCE)

#include "MediaPlayerPrivate.h"
#include <wtf/Logger.h>
#include <wtf/MediaTime.h>
#include <wtf/RefCounted.h>
#include <wtf/WeakPtr.h>

namespace WebCore {

class MediaSource;
class MockMediaSourcePrivate;

class MockMediaPlayerMediaSource final
    : public MediaPlayerPrivateInterface
    , public RefCounted<MockMediaPlayerMediaSource>
    , public CanMakeWeakPtr<MockMediaPlayerMediaSource> {
public:
    void ref() const final { RefCounted::ref(); }
    void deref() const final { RefCounted::deref(); }

    explicit MockMediaPlayerMediaSource(MediaPlayer*);

    // MediaPlayer Engine Support
    WEBCORE_EXPORT static void registerMediaEngine(MediaEngineRegistrar);
    static void getSupportedTypes(HashSet<String>& types);
    static MediaPlayer::SupportsType supportsType(const MediaEngineSupportParameters&);

    virtual ~MockMediaPlayerMediaSource();

    constexpr MediaPlayerType mediaPlayerType() const final { return MediaPlayerType::MockMSE; }

    void advanceCurrentTime();
    MediaTime currentTime() const override;
    bool timeIsProgressing() const override;
    void notifyActiveSourceBuffersChanged() final;
    void updateDuration(const MediaTime&);

    MediaPlayer::ReadyState readyState() const override;
    void setReadyState(MediaPlayer::ReadyState);
    void setNetworkState(MediaPlayer::NetworkState);

#if !RELEASE_LOG_DISABLED
    uint64_t mediaPlayerLogIdentifier() { return m_player.get()->mediaPlayerLogIdentifier(); }
    const Logger& mediaPlayerLogger() { return m_player.get()->mediaPlayerLogger(); }
#endif

private:
    // MediaPlayerPrivate Overrides
    void load(const String& url) override;
    void load(const URL&, const ContentType&, MediaSourcePrivateClient&) override;
#if ENABLE(MEDIA_STREAM)
    void load(MediaStreamPrivate&) override { }
#endif
    void cancelLoad() override;
    void play() override;
    void pause() override;
    FloatSize naturalSize() const override;
    bool hasVideo() const override;
    bool hasAudio() const override;
    void setPageIsVisible(bool) final;
    void seekToTarget(const SeekTarget&) final;
    bool seeking() const final;
    bool paused() const override;
    MediaPlayer::NetworkState networkState() const override;
    MediaTime maxTimeSeekable() const override;
    const PlatformTimeRanges& buffered() const override;
    bool didLoadingProgress() const override;
    void setPresentationSize(const IntSize&) override;
    void paint(GraphicsContext&, const FloatRect&) override;
    MediaTime duration() const override;
    std::optional<VideoPlaybackQualityMetrics> videoPlaybackQualityMetrics() override;
    DestinationColorSpace colorSpace() override;

    ThreadSafeWeakPtr<MediaPlayer> m_player;
    RefPtr<MockMediaSourcePrivate> m_mediaSourcePrivate;

    MediaTime m_currentTime;
    MediaTime m_duration;
    std::optional<SeekTarget> m_lastSeekTarget;
    MediaPlayer::ReadyState m_readyState { MediaPlayer::ReadyState::HaveNothing };
    MediaPlayer::NetworkState m_networkState { MediaPlayer::NetworkState::Empty };
    bool m_playing { false };
};

}

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::MockMediaPlayerMediaSource)
static bool isType(const WebCore::MediaPlayerPrivateInterface& player) { return player.mediaPlayerType() == WebCore::MediaPlayerType::MockMSE; }
SPECIALIZE_TYPE_TRAITS_END()

#endif // ENABLE(MEDIA_SOURCE)

#endif // MockMediaPlayerMediaSource_h

