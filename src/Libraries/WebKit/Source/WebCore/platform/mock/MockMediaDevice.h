/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 27, 2023.
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

#if ENABLE(MEDIA_STREAM)

#include "CaptureDevice.h"
#include "RealtimeMediaSource.h"
#include "VideoPreset.h"

namespace WebCore {

struct MockMicrophoneProperties {
    int defaultSampleRate { 44100 };
    std::optional<bool> echoCancellation;
};

struct MockSpeakerProperties {
    String relatedMicrophoneId;
    int defaultSampleRate { 44100 };
};

// FIXME: Add support for other properties.
struct MockCameraProperties {
    double defaultFrameRate { 30 };
    VideoFacingMode facingMode { VideoFacingMode::User };
    Vector<VideoPresetData> presets { { { 640, 480 }, { { 30, 30 }, { 15, 15 } }, 1, 2, false } };
    Color fillColor { Color::black };
    Vector<MeteringMode> whiteBalanceMode { MeteringMode::None };
    bool hasTorch { false };
    bool hasBackgroundBlur { false };
};

struct MockDisplayProperties {
    CaptureDevice::DeviceType type;
    Color fillColor { Color::lightGray };
    IntSize defaultSize;
};

struct MockMediaDevice {
    bool isMicrophone() const { return std::holds_alternative<MockMicrophoneProperties>(properties); }
    bool isSpeaker() const { return std::holds_alternative<MockSpeakerProperties>(properties); }
    bool isCamera() const { return std::holds_alternative<MockCameraProperties>(properties); }
    bool isDisplay() const { return std::holds_alternative<MockDisplayProperties>(properties); }

    enum class Flag : uint8_t {
        Ephemeral   = 1 << 0,
        Invalid     = 1 << 1,
    };
    using Flags = OptionSet<Flag>;

    CaptureDevice captureDevice() const
    {
        if (isMicrophone())
            return CaptureDevice { persistentId, CaptureDevice::DeviceType::Microphone, label, persistentId, true, false, true, flags.contains(Flag::Ephemeral) };

        if (isSpeaker())
            return CaptureDevice { persistentId, CaptureDevice::DeviceType::Speaker, label, speakerProperties()->relatedMicrophoneId, true, false, true, flags.contains(Flag::Ephemeral) };

        if (isCamera())
            return CaptureDevice { persistentId, CaptureDevice::DeviceType::Camera, label, persistentId, true, false, true, flags.contains(Flag::Ephemeral) };

        ASSERT(isDisplay());
        return CaptureDevice { persistentId, std::get<MockDisplayProperties>(properties).type, label, emptyString(), true, false, true, flags.contains(Flag::Ephemeral) };
    }

    CaptureDevice::DeviceType type() const
    {
        if (isMicrophone())
            return CaptureDevice::DeviceType::Microphone;
        if (isSpeaker())
            return CaptureDevice::DeviceType::Speaker;
        if (isCamera())
            return CaptureDevice::DeviceType::Camera;

        ASSERT(isDisplay());
        return std::get<MockDisplayProperties>(properties).type;
    }

    const MockSpeakerProperties* speakerProperties() const
    {
        return isSpeaker() ? &std::get<MockSpeakerProperties>(properties) : nullptr;
    }

    const MockCameraProperties* cameraProperties() const
    {
        return isCamera() ? &std::get<MockCameraProperties>(properties) : nullptr;
    }

    String persistentId;
    String label;
    Flags flags;
    std::variant<MockMicrophoneProperties, MockSpeakerProperties, MockCameraProperties, MockDisplayProperties> properties;
};

} // namespace WebCore

#endif // ENABLE(MEDIA_STREAM)
