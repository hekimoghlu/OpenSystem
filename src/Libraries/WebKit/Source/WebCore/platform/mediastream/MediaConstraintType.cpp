/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 3, 2024.
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
#include "MediaConstraintType.h"

#if ENABLE(MEDIA_STREAM)

namespace WebCore {

String convertToString(MediaConstraintType type)
{
    switch (type) {
    case MediaConstraintType::Unknown:
        return ""_s;
    case MediaConstraintType::Width:
        return "width"_s;
    case MediaConstraintType::Height:
        return "height"_s;
    case MediaConstraintType::AspectRatio:
        return "aspectRatio"_s;
    case MediaConstraintType::FrameRate:
        return "frameRate"_s;
    case MediaConstraintType::FacingMode:
        return "facingMode"_s;
    case MediaConstraintType::Volume:
        return "volume"_s;
    case MediaConstraintType::SampleRate:
        return "sampleRate"_s;
    case MediaConstraintType::SampleSize:
        return "sampleSize"_s;
    case MediaConstraintType::EchoCancellation:
        return "echoCancellation"_s;
    case MediaConstraintType::DeviceId:
        return "deviceId"_s;
    case MediaConstraintType::GroupId:
        return "groupId"_s;
    case MediaConstraintType::DisplaySurface:
        return "displaySurface"_s;
    case MediaConstraintType::LogicalSurface:
        return "logicalSurface"_s;
    case MediaConstraintType::FocusDistance:
        return "focusDistance"_s;
    case MediaConstraintType::WhiteBalanceMode:
        return "whiteBalanceMode"_s;
    case MediaConstraintType::Zoom:
        return "zoom"_s;
    case MediaConstraintType::Torch:
        return "torch"_s;
    case MediaConstraintType::BackgroundBlur:
        return "backgroundBlur"_s;
    case MediaConstraintType::PowerEfficient:
        return "powerEfficient"_s;
    }

    ASSERT_NOT_REACHED();
    return { };
}

} // namespace WebCore

#endif // ENABLE(MEDIA_STREAM)
