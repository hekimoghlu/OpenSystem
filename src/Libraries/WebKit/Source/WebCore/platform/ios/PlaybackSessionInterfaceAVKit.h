/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 29, 2024.
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

#if HAVE(AVKIT_CONTENT_SOURCE)

#include "PlaybackSessionInterfaceIOS.h"
#include <wtf/TZoneMalloc.h>

OBJC_CLASS WebAVContentSource;

namespace WebCore {

class PlaybackSessionInterfaceAVKit final : public PlaybackSessionInterfaceIOS {
    WTF_MAKE_TZONE_ALLOCATED(PlaybackSessionInterfaceAVKit);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(PlaybackSessionInterfaceAVKit);
public:
    WEBCORE_EXPORT static Ref<PlaybackSessionInterfaceAVKit> create(PlaybackSessionModel&);
    ~PlaybackSessionInterfaceAVKit();

    void nowPlayingMetadataChanged(const NowPlayingMetadata&);

    // PlaybackSessionInterfaceIOS overrides
    WebAVPlayerController *playerController() const final { return nullptr; }
    WKSLinearMediaPlayer *linearMediaPlayer() const final { return nullptr; }
    void durationChanged(double) final;
    void currentTimeChanged(double, double) final;
    void bufferedTimeChanged(double) final { }
    void rateChanged(OptionSet<PlaybackSessionModel::PlaybackState>, double, double) final;
    void seekableRangesChanged(const TimeRanges&, double, double) final;
    void canPlayFastReverseChanged(bool) final;
    void audioMediaSelectionOptionsChanged(const Vector<MediaSelectionOption>&, uint64_t) final;
    void legibleMediaSelectionOptionsChanged(const Vector<MediaSelectionOption>&, uint64_t) final;
    void audioMediaSelectionIndexChanged(uint64_t) final;
    void legibleMediaSelectionIndexChanged(uint64_t) final;
    void externalPlaybackChanged(bool, PlaybackSessionModel::ExternalPlaybackTargetType, const String&) final { }
    void wirelessVideoPlaybackDisabledChanged(bool) final { }
    void mutedChanged(bool) final;
    void volumeChanged(double) final;
    void startObservingNowPlayingMetadata() final;
    void stopObservingNowPlayingMetadata() final;
#if !RELEASE_LOG_DISABLED
    ASCIILiteral logClassName() const final;
#endif

private:
    PlaybackSessionInterfaceAVKit(PlaybackSessionModel&);

    RetainPtr<WebAVContentSource> m_contentSource;
    NowPlayingMetadataObserver m_nowPlayingMetadataObserver;
};

} // namespace WebCore

#endif // HAVE(AVKIT_CONTENT_SOURCE)
