/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 14, 2025.
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

#include "MediaSessionIdentifier.h"
#include <wtf/URL.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

struct MediaUsageInfo {
    URL mediaURL;
    bool hasSource { false };
    bool isPlaying { false };
    bool canShowControlsManager { false };
    bool canShowNowPlayingControls { false };
    bool isSuspended { false };
    bool isInActiveDocument { false };
    bool isFullscreen { false };
    bool isMuted { false };
    bool isMediaDocumentInMainFrame { false };
    bool isVideo { false };
    bool isAudio { false };
    bool hasVideo { false };
    bool hasAudio { false };
    bool hasRenderer { false };
    bool audioElementWithUserGesture { false };
    bool userHasPlayedAudioBefore { false };
    bool isElementRectMostlyInMainFrame { false };
    bool playbackPermitted { false };
    bool pageMediaPlaybackSuspended { false };
    bool isMediaDocumentAndNotOwnerElement { false };
    bool pageExplicitlyAllowsElementToAutoplayInline { false };
    bool requiresFullscreenForVideoPlaybackAndFullscreenNotPermitted { false };
    bool isVideoAndRequiresUserGestureForVideoRateChange { false };
    bool isAudioAndRequiresUserGestureForAudioRateChange { false };
    bool isVideoAndRequiresUserGestureForVideoDueToLowPowerMode { false };
    bool isVideoAndRequiresUserGestureForVideoDueToAggressiveThermalMitigation { false };
    bool noUserGestureRequired { false };
    bool requiresPlaybackAndIsNotPlaying { false };
    bool hasEverNotifiedAboutPlaying { false };
    bool outsideOfFullscreen { false };
    bool isLargeEnoughForMainContent { false };
#if PLATFORM(COCOA) && !HAVE(CGS_FIX_FOR_RADAR_97530095)
    bool isInViewport { false };
#endif

    friend bool operator==(const MediaUsageInfo&, const MediaUsageInfo&) = default;
};

} // namespace WebCore
