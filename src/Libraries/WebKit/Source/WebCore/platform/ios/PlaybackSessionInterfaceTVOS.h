/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 22, 2023.
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

#if PLATFORM(APPLETV)

#include "PlaybackSessionInterfaceIOS.h"

namespace WebCore {

class PlaybackSessionInterfaceTVOS final : public PlaybackSessionInterfaceIOS {
    WTF_MAKE_FAST_ALLOCATED;
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(PlaybackSessionInterfaceTVOS);
public:
    WEBCORE_EXPORT static Ref<PlaybackSessionInterfaceTVOS> create(PlaybackSessionModel&);

    // PlaybackSessionInterfaceIOS
    WebAVPlayerController *playerController() const final { return { }; }
    WKSLinearMediaPlayer *linearMediaPlayer() const final { return { }; }
    void durationChanged(double) final { }
    void currentTimeChanged(double, double) final { }
    void bufferedTimeChanged(double) final { }
    void rateChanged(OptionSet<PlaybackSessionModel::PlaybackState>, double, double) final { }
    void seekableRangesChanged(const TimeRanges&, double, double) final { }
    void canPlayFastReverseChanged(bool) final { }
    void audioMediaSelectionOptionsChanged(const Vector<MediaSelectionOption>&, uint64_t) final { }
    void legibleMediaSelectionOptionsChanged(const Vector<MediaSelectionOption>&, uint64_t) final { }
    void externalPlaybackChanged(bool, PlaybackSessionModel::ExternalPlaybackTargetType, const String&) final { }
    void wirelessVideoPlaybackDisabledChanged(bool) final { }
    void mutedChanged(bool) final { }
    void volumeChanged(double) final { }
#if !RELEASE_LOG_DISABLED
    ASCIILiteral logClassName() const final { return "PlaybackSessionInterfaceTVOS"_s; }
#endif

private:
    PlaybackSessionInterfaceTVOS(PlaybackSessionModel&);
};

} // namespace WebCore

#endif // PLATFORM(APPLETV)
