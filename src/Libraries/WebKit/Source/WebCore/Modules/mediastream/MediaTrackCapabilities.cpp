/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 1, 2025.
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
#include "config.h"
#include "MediaTrackCapabilities.h"

#if ENABLE(MEDIA_STREAM)

#include "JSMeteringMode.h"
#include "RealtimeMediaSourceCapabilities.h"

namespace WebCore {

static DoubleRange capabilityDoubleRange(const DoubleCapabilityRange& value)
{
    auto min = value.min();
    auto max = value.max();

    ASSERT(min != std::numeric_limits<double>::min() || max != std::numeric_limits<double>::max());

    DoubleRange range;
    if (min != std::numeric_limits<double>::min())
        range.min = min;
    if (max != std::numeric_limits<double>::max())
        range.max = max;

    return range;
}

static LongRange capabilityLongRange(const LongCapabilityRange& value)
{
    return { value.max(), value.min() };
}

static Vector<String> capabilityStringVector(const Vector<VideoFacingMode>& modes)
{
    return modes.map([](auto& mode) {
        return convertEnumerationToString(mode);
    });
}

static Vector<String> capabilityStringVector(const Vector<MeteringMode>& modes)
{
    return modes.map([](auto& mode) {
        return convertEnumerationToString(mode);
    });
}

static Vector<bool> capabilityBooleanVector(RealtimeMediaSourceCapabilities::EchoCancellation cancellation)
{
    Vector<bool> result;
    result.reserveInitialCapacity(2);
    switch (cancellation) {
    case RealtimeMediaSourceCapabilities::EchoCancellation::On:
        result.append(true);
        break;
    case RealtimeMediaSourceCapabilities::EchoCancellation::Off:
        result.append(false);
        break;
    case RealtimeMediaSourceCapabilities::EchoCancellation::OnOrOff:
        result.append(true);
        result.append(false);
        break;
    }
    return result;
}

static Vector<bool> capabilityBooleanVector(RealtimeMediaSourceCapabilities::BackgroundBlur backgroundBlur)
{
    Vector<bool> result;
    result.reserveInitialCapacity(2);
    switch (backgroundBlur) {
    case RealtimeMediaSourceCapabilities::BackgroundBlur::On:
        result.append(true);
        break;
    case RealtimeMediaSourceCapabilities::BackgroundBlur::Off:
        result.append(false);
        break;
    case RealtimeMediaSourceCapabilities::BackgroundBlur::OnOff:
        result.append(false);
        result.append(true);
        break;
    }
    return result;
}

static Vector<bool> powerEfficientCapabilityVector(bool powerEfficient)
{
    Vector<bool> result;
    result.reserveInitialCapacity(2);
    result.append(false);
    if (powerEfficient)
        result.append(true);

    return result;
}

MediaTrackCapabilities toMediaTrackCapabilities(const RealtimeMediaSourceCapabilities& capabilities)
{
    MediaTrackCapabilities result;
    if (capabilities.supportsWidth())
        result.width = capabilityLongRange(capabilities.width());
    if (capabilities.supportsHeight())
        result.height = capabilityLongRange(capabilities.height());
    if (capabilities.supportsAspectRatio())
        result.aspectRatio = capabilityDoubleRange(capabilities.aspectRatio());
    if (capabilities.supportsFrameRate())
        result.frameRate = capabilityDoubleRange(capabilities.frameRate());
    if (capabilities.supportsFacingMode())
        result.facingMode = capabilityStringVector(capabilities.facingMode());
    if (capabilities.supportsVolume())
        result.volume = capabilityDoubleRange(capabilities.volume());
    if (capabilities.supportsSampleRate())
        result.sampleRate = capabilityLongRange(capabilities.sampleRate());
    if (capabilities.supportsSampleSize())
        result.sampleSize = capabilityLongRange(capabilities.sampleSize());
    if (capabilities.supportsEchoCancellation())
        result.echoCancellation = capabilityBooleanVector(capabilities.echoCancellation());
    if (capabilities.supportsDeviceId())
        result.deviceId = capabilities.deviceId();
    if (capabilities.supportsGroupId())
        result.groupId = capabilities.groupId();
    if (capabilities.supportsFocusDistance())
        result.focusDistance = capabilityDoubleRange(capabilities.focusDistance());
    if (capabilities.supportsWhiteBalanceMode())
        result.whiteBalanceMode = capabilityStringVector(capabilities.whiteBalanceModes());
    if (capabilities.supportsZoom())
        result.zoom = capabilityDoubleRange(capabilities.zoom());
    if (capabilities.supportsTorch())
        result.torch = capabilities.torch();
    if (capabilities.supportsBackgroundBlur())
        result.backgroundBlur = capabilityBooleanVector(capabilities.backgroundBlur());

    if (capabilities.supportsPowerEfficient())
        result.powerEfficient = powerEfficientCapabilityVector(capabilities.powerEfficient());

    return result;
}

}

#endif
