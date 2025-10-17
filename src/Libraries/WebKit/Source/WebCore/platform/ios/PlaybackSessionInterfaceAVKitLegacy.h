/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 28, 2025.
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

#if PLATFORM(COCOA) && HAVE(AVKIT)

#include "PlaybackSessionInterfaceIOS.h"
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class WEBCORE_EXPORT PlaybackSessionInterfaceAVKitLegacy final : public PlaybackSessionInterfaceIOS {
    WTF_MAKE_TZONE_ALLOCATED_EXPORT(PlaybackSessionInterfaceAVKitLegacy, WEBCORE_EXPORT);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(PlaybackSessionInterfaceAVKitLegacy);
public:
    static Ref<PlaybackSessionInterfaceAVKitLegacy> create(PlaybackSessionModel&);
    ~PlaybackSessionInterfaceAVKitLegacy();
    void invalidate() final;

    WebAVPlayerController *playerController() const final;
    WKSLinearMediaPlayer *linearMediaPlayer() const final;
    void durationChanged(double) final;
    void currentTimeChanged(double currentTime, double anchorTime) final;
    void bufferedTimeChanged(double) final;
    void rateChanged(OptionSet<PlaybackSessionModel::PlaybackState>, double playbackRate, double defaultPlaybackRate) final;
    void seekableRangesChanged(const TimeRanges&, double lastModifiedTime, double liveUpdateInterval) final;
    void canPlayFastReverseChanged(bool) final;
    void audioMediaSelectionOptionsChanged(const Vector<MediaSelectionOption>& options, uint64_t selectedIndex) final;
    void legibleMediaSelectionOptionsChanged(const Vector<MediaSelectionOption>& options, uint64_t selectedIndex) final;
    void externalPlaybackChanged(bool enabled, PlaybackSessionModel::ExternalPlaybackTargetType, const String& localizedDeviceName) final;
    void wirelessVideoPlaybackDisabledChanged(bool) final;
    void mutedChanged(bool) final;
    void volumeChanged(double) final;
#if !RELEASE_LOG_DISABLED
    ASCIILiteral logClassName() const final;
#endif

private:
    PlaybackSessionInterfaceAVKitLegacy(PlaybackSessionModel&);

    RetainPtr<WebAVPlayerController> m_playerController;

};

} // namespace WebCore

#endif // PLATFORM(COCOA) && HAVE(AVKIT)
