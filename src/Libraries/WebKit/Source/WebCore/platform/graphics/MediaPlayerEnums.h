/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 10, 2021.
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

#include <wtf/text/WTFString.h>

namespace WebCore {

enum class MediaPlayerNetworkState : uint8_t {
    Empty,
    Idle,
    Loading,
    Loaded,
    FormatError,
    NetworkError,
    DecodeError
};

enum class MediaPlayerReadyState : uint8_t {
    HaveNothing,
    HaveMetadata,
    HaveCurrentData,
    HaveFutureData,
    HaveEnoughData
};

enum class MediaPlayerMovieLoadType : uint8_t {
    Unknown,
    Download,
    StoredStream,
    LiveStream,
    HttpLiveStream,
};

enum class MediaPlayerPreload : uint8_t {
    None,
    MetaData,
    Auto
};

enum class MediaPlayerVideoGravity : uint8_t {
    Resize,
    ResizeAspect,
    ResizeAspectFill
};

enum class MediaPlayerSupportsType : uint8_t {
    IsNotSupported,
    IsSupported,
    MayBeSupported
};

enum class MediaPlayerBufferingPolicy : uint8_t {
    Default,
    LimitReadAhead,
    MakeResourcesPurgeable,
    PurgeResources,
};

enum class MediaPlayerMediaEngineIdentifier : uint8_t {
    AVFoundation,
    AVFoundationMSE,
    AVFoundationMediaStream,
    AVFoundationCF,
    GStreamer,
    GStreamerMSE,
    HolePunch,
    MediaFoundation,
    MockMSE,
    CocoaWebM
};

enum class MediaPlayerWirelessPlaybackTargetType : uint8_t {
    TargetTypeNone,
    TargetTypeAirPlay,
    TargetTypeTVOut
};

enum class MediaPlayerPitchCorrectionAlgorithm : uint8_t {
    BestAllAround,
    BestForMusic,
    BestForSpeech,
};

enum class MediaPlayerNeedsRenderingModeChanged : bool {
    No,
    Yes,
};

class MediaPlayerEnums {
public:
    using NetworkState = MediaPlayerNetworkState;
    using ReadyState = MediaPlayerReadyState;
    using MovieLoadType = MediaPlayerMovieLoadType;
    using Preload = MediaPlayerPreload;
    using VideoGravity = MediaPlayerVideoGravity;
    using SupportsType = MediaPlayerSupportsType;
    using BufferingPolicy = MediaPlayerBufferingPolicy;
    using MediaEngineIdentifier = MediaPlayerMediaEngineIdentifier;
    using WirelessPlaybackTargetType = MediaPlayerWirelessPlaybackTargetType;
    using PitchCorrectionAlgorithm = MediaPlayerPitchCorrectionAlgorithm;
    using NeedsRenderingModeChanged = MediaPlayerNeedsRenderingModeChanged;

    enum {
        VideoFullscreenModeNone = 0,
        VideoFullscreenModeStandard = 1 << 0,
        VideoFullscreenModePictureInPicture = 1 << 1,
        VideoFullscreenModeInWindow = 1 << 2,
        VideoFullscreenModeAllValidBitsMask = (VideoFullscreenModeStandard | VideoFullscreenModePictureInPicture | VideoFullscreenModeInWindow)
    };
    typedef uint32_t VideoFullscreenMode;
};

WEBCORE_EXPORT WTF::TextStream& operator<<(WTF::TextStream&, MediaPlayerEnums::VideoGravity);
WEBCORE_EXPORT String convertEnumerationToString(MediaPlayerEnums::ReadyState);
String convertEnumerationToString(MediaPlayerEnums::NetworkState);
String convertEnumerationToString(MediaPlayerEnums::Preload);
String convertEnumerationToString(MediaPlayerEnums::SupportsType);
String convertEnumerationToString(MediaPlayerEnums::BufferingPolicy);

} // namespace WebCore


namespace WTF {

template<typename Type>
struct LogArgument;

template <>
struct LogArgument<WebCore::MediaPlayerEnums::ReadyState> {
    static String toString(const WebCore::MediaPlayerEnums::ReadyState state)
    {
        return convertEnumerationToString(state);
    }
};

template <>
struct LogArgument<WebCore::MediaPlayerEnums::NetworkState> {
    static String toString(const WebCore::MediaPlayerEnums::NetworkState state)
    {
        return convertEnumerationToString(state);
    }
};

template <>
struct LogArgument<WebCore::MediaPlayerEnums::BufferingPolicy> {
    static String toString(const WebCore::MediaPlayerEnums::BufferingPolicy policy)
    {
        return convertEnumerationToString(policy);
    }
};

}; // namespace WTF
