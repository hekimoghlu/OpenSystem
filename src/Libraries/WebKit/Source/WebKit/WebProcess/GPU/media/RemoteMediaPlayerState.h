/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 25, 2024.
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

#if ENABLE(GPU_PROCESS) && ENABLE(VIDEO)

#include <WebCore/FloatSize.h>
#include <WebCore/MediaPlayerEnums.h>
#include <WebCore/PlatformTimeRanges.h>
#include <WebCore/VideoPlaybackQualityMetrics.h>
#include <wtf/MediaTime.h>
#include <wtf/MonotonicTime.h>

namespace WebKit {

struct RemoteMediaPlayerState {
    MediaTime duration;
    MediaTime minTimeSeekable;
    MediaTime maxTimeSeekable;
    MediaTime startDate;
    MediaTime startTime;
    String languageOfPrimaryAudioTrack;
    String wirelessPlaybackTargetName;
    std::optional<WebCore::PlatformTimeRanges> bufferedRanges;
    WebCore::MediaPlayerEnums::NetworkState networkState { WebCore::MediaPlayerEnums::NetworkState::Empty };
    WebCore::MediaPlayerEnums::MovieLoadType movieLoadType { WebCore::MediaPlayerEnums::MovieLoadType::Unknown };
    WebCore::MediaPlayerEnums::WirelessPlaybackTargetType wirelessPlaybackTargetType { WebCore::MediaPlayerEnums::WirelessPlaybackTargetType::TargetTypeNone };
    WebCore::FloatSize naturalSize;
    double maxFastForwardRate { 0 };
    double minFastReverseRate { 0 };
    double seekableTimeRangesLastModifiedTime { 0 };
    double liveUpdateInterval { 0 };
    uint64_t totalBytes { 0 };
    std::optional<WebCore::VideoPlaybackQualityMetrics> videoMetrics;
    std::optional<bool> documentIsCrossOrigin { true };
    bool paused { true };
    bool canSaveMediaData { false };
    bool hasAudio { false };
    bool hasVideo { false };
    bool hasClosedCaptions { false };
    bool hasAvailableVideoFrame { false };
    bool wirelessVideoPlaybackDisabled { false };
    bool didPassCORSAccessCheck { false };
};

} // namespace WebKit

#endif // ENABLE(GPU_PROCESS) && ENABLE(VIDEO)
