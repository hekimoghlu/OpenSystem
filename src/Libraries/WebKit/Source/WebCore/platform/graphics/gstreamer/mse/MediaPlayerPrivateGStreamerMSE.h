/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 26, 2022.
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

#if ENABLE(VIDEO) && USE(GSTREAMER) && ENABLE(MEDIA_SOURCE)

#include "GStreamerCommon.h"
#include "MediaPlayerPrivateGStreamer.h"
#include "MediaSample.h"
#include "MediaSourcePrivateGStreamer.h"

struct WebKitMediaSrc;

namespace WebCore {

class AppendPipeline;
class TrackQueue;
class MediaSourceTrackGStreamer;

class MediaPlayerPrivateGStreamerMSE : public MediaPlayerPrivateGStreamer {

public:
    Ref<MediaPlayerPrivateGStreamerMSE> create(MediaPlayer* player) { return adoptRef(*new MediaPlayerPrivateGStreamerMSE(player)); }
    virtual ~MediaPlayerPrivateGStreamerMSE();

    static void registerMediaEngine(MediaEngineRegistrar);

    constexpr MediaPlayerType mediaPlayerType() const final { return MediaPlayerType::GStreamerMSE; }

    void load(const String&) override;
    void load(const URL&, const ContentType&, MediaSourcePrivateClient&) override;

    void updateDownloadBufferingFlag() override { };

    void play() override;
    void pause() override;
    void seekToTarget(const SeekTarget&) override;
    bool doSeek(const SeekTarget&, float rate) override;

    void updatePipelineState(GstState);

    void durationChanged() override;
    MediaTime duration() const override;

    const PlatformTimeRanges& buffered() const override;
    MediaTime maxTimeSeekable() const override;
    bool timeIsProgressing() const override;
    void notifyActiveSourceBuffersChanged() final;

    void sourceSetup(GstElement*) override;

    // return false to avoid false-positive "stalled" event - it should be soon addressed in the spec
    // see: https://github.com/w3c/media-source/issues/88
    // see: https://w3c.github.io/media-source/#h-note-19
    bool supportsProgressMonitoring() const override { return false; }

    void setNetworkState(MediaPlayer::NetworkState);
    void setReadyState(MediaPlayer::ReadyState);

    void setInitialVideoSize(const FloatSize&);

    void didPreroll() override;

    void startSource(const Vector<RefPtr<MediaSourceTrackGStreamer>>& tracks);
    WebKitMediaSrc* webKitMediaSrc() { return reinterpret_cast<WebKitMediaSrc*>(m_source.get()); }

    void setEosWithNoBuffers(bool);

#if !RELEASE_LOG_DISABLED
    WTFLogChannel& logChannel() const final { return WebCore::LogMediaSource; }
#endif

    void checkPlayingConsistency() final;
#ifndef GST_DISABLE_GST_DEBUG
    void setShouldDisableSleep(bool) final;
#endif

private:
    explicit MediaPlayerPrivateGStreamerMSE(MediaPlayer*);

    friend class MediaPlayerFactoryGStreamerMSE;
    static void getSupportedTypes(HashSet<String>&);
    static MediaPlayer::SupportsType supportsType(const MediaEngineSupportParameters&);

    friend class AppendPipeline;
    friend class SourceBufferPrivateGStreamer;
    friend class MediaSourcePrivateGStreamer;

    size_t extraMemoryCost() const override;

    void updateStates() override;

    // FIXME: Implement videoPlaybackQualityMetrics.
    bool isTimeBuffered(const MediaTime&) const;

    bool isMediaSource() const override { return true; }

    void propagateReadyStateToPlayer();

    RefPtr<MediaSourcePrivateGStreamer> m_mediaSourcePrivate;
    MediaTime m_mediaTimeDuration { MediaTime::invalidTime() };
    Vector<RefPtr<MediaSourceTrackGStreamer>> m_tracks;

    bool m_isWaitingForPreroll = true;
    bool m_isEosWithNoBuffers = false;
    MediaPlayer::ReadyState m_mediaSourceReadyState = MediaPlayer::ReadyState::HaveNothing;
    MediaPlayer::NetworkState m_mediaSourceNetworkState = MediaPlayer::NetworkState::Empty;

    bool m_playbackStateChangedNotificationPending { false };
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::MediaPlayerPrivateGStreamerMSE)
static bool isType(const WebCore::MediaPlayerPrivateInterface& player) { return player.mediaPlayerType() == WebCore::MediaPlayerType::GStreamerMSE; }
SPECIALIZE_TYPE_TRAITS_END()

#endif // USE(GSTREAMER)
