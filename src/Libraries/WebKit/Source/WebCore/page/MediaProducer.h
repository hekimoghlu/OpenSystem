/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 10, 2023.
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

#include <wtf/OptionSet.h>
#include <wtf/WeakPtr.h>

namespace WebCore {
class MediaProducer;
}

namespace WTF {
template<typename T> struct IsDeprecatedWeakRefSmartPointerException;
template<> struct IsDeprecatedWeakRefSmartPointerException<WebCore::MediaProducer> : std::true_type { };
}

namespace WebCore {

enum class MediaProducerMediaState : uint32_t {
    IsPlayingAudio = 1 << 0,
    IsPlayingVideo = 1 << 1,
    IsPlayingToExternalDevice = 1 << 2,
    RequiresPlaybackTargetMonitoring = 1 << 3,
    ExternalDeviceAutoPlayCandidate = 1 << 4,
    DidPlayToEnd = 1 << 5,
    IsSourceElementPlaying = 1 << 6,
    IsNextTrackControlEnabled = 1 << 7,
    IsPreviousTrackControlEnabled = 1 << 8,
    HasPlaybackTargetAvailabilityListener = 1 << 9,
    HasAudioOrVideo = 1 << 10,
    HasActiveAudioCaptureDevice = 1 << 11,
    HasActiveVideoCaptureDevice = 1 << 12,
    HasMutedAudioCaptureDevice = 1 << 13,
    HasMutedVideoCaptureDevice = 1 << 14,
    HasInterruptedAudioCaptureDevice = 1 << 15,
    HasInterruptedVideoCaptureDevice = 1 << 16,
    HasUserInteractedWithMediaElement = 1 << 17,
    HasActiveScreenCaptureDevice = 1 << 18,
    HasMutedScreenCaptureDevice = 1 << 19,
    HasInterruptedScreenCaptureDevice = 1 << 20,
    HasActiveWindowCaptureDevice = 1 << 21,
    HasMutedWindowCaptureDevice = 1 << 22,
    HasInterruptedWindowCaptureDevice = 1 << 23,
    HasActiveSystemAudioCaptureDevice = 1 << 24,
    HasMutedSystemAudioCaptureDevice = 1 << 25,
    HasInterruptedSystemAudioCaptureDevice = 1 << 26,
    HasStreamingActivity = 1 << 27,
};
using MediaProducerMediaStateFlags = OptionSet<MediaProducerMediaState>;

enum class MediaProducerMediaCaptureKind : uint8_t {
    Microphone = 1 << 0,
    Camera = 1 << 1,
    Display = 1 << 2,
    SystemAudio = 1 << 3,
    EveryKind = 1 << 4,
};

enum class MediaProducerMutedState : uint8_t {
    AudioIsMuted = 1 << 0,
    AudioCaptureIsMuted = 1 << 1,
    VideoCaptureIsMuted = 1 << 2,
    ScreenCaptureIsMuted = 1 << 3,
    WindowCaptureIsMuted = 1 << 4,
    SystemAudioCaptureIsMuted = 1 << 5,
};
using MediaProducerMutedStateFlags = OptionSet<MediaProducerMutedState>;

class MediaProducer : public CanMakeWeakPtr<MediaProducer> {
public:
    using MediaState = MediaProducerMediaState;
    using MutedState = MediaProducerMutedState;
    using MediaStateFlags = MediaProducerMediaStateFlags;
    using MutedStateFlags = MediaProducerMutedStateFlags;

    static constexpr MediaStateFlags IsNotPlaying = { };
    static constexpr MediaStateFlags MicrophoneCaptureMask = { MediaState::HasActiveAudioCaptureDevice, MediaState::HasMutedAudioCaptureDevice, MediaState::HasInterruptedAudioCaptureDevice };
    static constexpr MediaStateFlags VideoCaptureMask = { MediaState::HasActiveVideoCaptureDevice, MediaState::HasMutedVideoCaptureDevice, MediaState::HasInterruptedVideoCaptureDevice };

    static constexpr MediaStateFlags ScreenCaptureMask = { MediaState::HasActiveScreenCaptureDevice, MediaState::HasMutedScreenCaptureDevice, MediaState::HasInterruptedScreenCaptureDevice };
    static constexpr MediaStateFlags WindowCaptureMask = { MediaState::HasActiveWindowCaptureDevice, MediaState::HasMutedWindowCaptureDevice, MediaState::HasInterruptedWindowCaptureDevice };
    static constexpr MediaStateFlags ActiveDisplayCaptureMask = { MediaState::HasActiveScreenCaptureDevice, MediaState::HasActiveWindowCaptureDevice };
    static constexpr MediaStateFlags MutedDisplayCaptureMask = { MediaState::HasMutedScreenCaptureDevice, MediaState::HasMutedWindowCaptureDevice };
    static constexpr MediaStateFlags DisplayCaptureMask = { ActiveDisplayCaptureMask | MutedDisplayCaptureMask };

    static constexpr MediaStateFlags SystemAudioCaptureMask = { MediaState::HasActiveSystemAudioCaptureDevice, MediaState::HasMutedSystemAudioCaptureDevice, MediaState::HasInterruptedSystemAudioCaptureDevice };

    static constexpr MediaStateFlags ActiveCaptureMask = { MediaState::HasActiveAudioCaptureDevice, MediaState::HasActiveVideoCaptureDevice, MediaState::HasActiveScreenCaptureDevice, MediaState::HasActiveWindowCaptureDevice, MediaState::HasActiveSystemAudioCaptureDevice };
    static constexpr MediaStateFlags MutedCaptureMask = { MediaState::HasMutedAudioCaptureDevice, MediaState::HasMutedVideoCaptureDevice, MediaState::HasMutedScreenCaptureDevice, MediaState::HasMutedWindowCaptureDevice, MediaState::HasMutedSystemAudioCaptureDevice };

    static constexpr MediaStateFlags MediaCaptureMask = {
        MediaState::HasActiveAudioCaptureDevice,
        MediaState::HasMutedAudioCaptureDevice,
        MediaState::HasInterruptedAudioCaptureDevice,
        MediaState::HasActiveVideoCaptureDevice,
        MediaState::HasMutedVideoCaptureDevice,
        MediaState::HasInterruptedVideoCaptureDevice,
        MediaState::HasActiveScreenCaptureDevice,
        MediaState::HasMutedScreenCaptureDevice,
        MediaState::HasInterruptedScreenCaptureDevice,
        MediaState::HasActiveWindowCaptureDevice,
        MediaState::HasMutedWindowCaptureDevice,
        MediaState::HasInterruptedWindowCaptureDevice,
        MediaState::HasActiveSystemAudioCaptureDevice,
        MediaState::HasMutedSystemAudioCaptureDevice,
        MediaState::HasInterruptedSystemAudioCaptureDevice
    };
    static constexpr MediaStateFlags IsCapturingAudioMask = { MicrophoneCaptureMask | SystemAudioCaptureMask };
    static constexpr MediaStateFlags IsCapturingVideoMask = { VideoCaptureMask | DisplayCaptureMask };

    static bool isCapturing(MediaStateFlags state) { return state.containsAny(ActiveCaptureMask) || state.containsAny(MutedCaptureMask); }

    virtual MediaStateFlags mediaState() const = 0;

    static constexpr MutedStateFlags AudioAndVideoCaptureIsMuted = { MutedState::AudioCaptureIsMuted, MutedState::VideoCaptureIsMuted };
    static constexpr MutedStateFlags MediaStreamCaptureIsMuted = { MutedState::AudioCaptureIsMuted, MutedState::VideoCaptureIsMuted, MutedState::ScreenCaptureIsMuted, MutedState::WindowCaptureIsMuted, MutedState::SystemAudioCaptureIsMuted };

    virtual void visibilityAdjustmentStateDidChange() { }
    virtual void pageMutedStateDidChange() = 0;

protected:
    virtual ~MediaProducer() = default;
};

} // namespace WebCore
