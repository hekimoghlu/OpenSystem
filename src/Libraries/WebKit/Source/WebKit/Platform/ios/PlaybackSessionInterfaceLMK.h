/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 12, 2022.
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

#if ENABLE(LINEAR_MEDIA_PLAYER)

#include <WebCore/NowPlayingMetadataObserver.h>
#include <WebCore/PlaybackSessionInterfaceIOS.h>
#include <wtf/Observer.h>
#include <wtf/TZoneMalloc.h>

OBJC_CLASS WKLinearMediaPlayerDelegate;

namespace WebKit {

class PlaybackSessionInterfaceLMK final : public WebCore::PlaybackSessionInterfaceIOS {
    WTF_MAKE_TZONE_ALLOCATED(PlaybackSessionInterfaceLMK);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(PlaybackSessionInterfaceLMK);
public:
    static Ref<PlaybackSessionInterfaceLMK> create(WebCore::PlaybackSessionModel&);
    ~PlaybackSessionInterfaceLMK();

    WebAVPlayerController *playerController() const final { return nullptr; }
    WKSLinearMediaPlayer *linearMediaPlayer() const final;
    void durationChanged(double) final;
    void currentTimeChanged(double, double) final;
    void bufferedTimeChanged(double) final { }
    void rateChanged(OptionSet<WebCore::PlaybackSessionModel::PlaybackState>, double, double) final;
    void seekableRangesChanged(const WebCore::TimeRanges&, double, double) final;
    void canPlayFastReverseChanged(bool) final;
    void audioMediaSelectionOptionsChanged(const Vector<WebCore::MediaSelectionOption>&, uint64_t) final;
    void legibleMediaSelectionOptionsChanged(const Vector<WebCore::MediaSelectionOption>&, uint64_t) final;
    void audioMediaSelectionIndexChanged(uint64_t) final;
    void legibleMediaSelectionIndexChanged(uint64_t) final;
    void externalPlaybackChanged(bool, WebCore::PlaybackSessionModel::ExternalPlaybackTargetType, const String&) final { }
    void wirelessVideoPlaybackDisabledChanged(bool) final { }
    void mutedChanged(bool) final;
    void volumeChanged(double) final;
    void supportsLinearMediaPlayerChanged(bool) final;
    void spatialVideoMetadataChanged(const std::optional<WebCore::SpatialVideoMetadata>&) final;
    void startObservingNowPlayingMetadata() final;
    void stopObservingNowPlayingMetadata() final;
#if !RELEASE_LOG_DISABLED
    ASCIILiteral logClassName() const final;
#endif

    void nowPlayingMetadataChanged(const WebCore::NowPlayingMetadata&);

    void setSpatialVideoEnabled(bool enabled) { m_spatialVideoEnabled = enabled; }
    bool spatialVideoEnabled() const { return m_spatialVideoEnabled; }

    void swapFullscreenModesWith(PlaybackSessionInterfaceIOS&);

private:
    PlaybackSessionInterfaceLMK(WebCore::PlaybackSessionModel&);

    RetainPtr<WKSLinearMediaPlayer> m_player;
    RetainPtr<WKLinearMediaPlayerDelegate> m_playerDelegate;
    WebCore::NowPlayingMetadataObserver m_nowPlayingMetadataObserver;
    bool m_spatialVideoEnabled { false };
    WebCore::VideoReceiverEndpoint m_videoReceiverEndpoint;
};

} // namespace WebKit

#endif // ENABLE(LINEAR_MEDIA_PLAYER)
