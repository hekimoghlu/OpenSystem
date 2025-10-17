/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 26, 2022.
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
#include "RealtimeMediaSourceSupportedConstraints.h"

#if ENABLE(MEDIA_STREAM)

namespace WebCore {

bool RealtimeMediaSourceSupportedConstraints::supportsConstraint(MediaConstraintType constraint) const
{
    switch (constraint) {
    case MediaConstraintType::Unknown:
        return false;
    case MediaConstraintType::Width:
        return supportsWidth();
    case MediaConstraintType::Height:
        return supportsHeight();
    case MediaConstraintType::AspectRatio:
        return supportsAspectRatio();
    case MediaConstraintType::FrameRate:
        return supportsFrameRate();
    case MediaConstraintType::FacingMode:
        return supportsFacingMode();
    case MediaConstraintType::Volume:
        return supportsVolume();
    case MediaConstraintType::SampleRate:
        return supportsSampleRate();
    case MediaConstraintType::SampleSize:
        return supportsSampleSize();
    case MediaConstraintType::EchoCancellation:
        return supportsEchoCancellation();
    case MediaConstraintType::DeviceId:
        return supportsDeviceId();
    case MediaConstraintType::GroupId:
        return supportsGroupId();
    case MediaConstraintType::DisplaySurface:
        return supportsDisplaySurface();
    case MediaConstraintType::LogicalSurface:
        return supportsLogicalSurface();
    case MediaConstraintType::FocusDistance:
        return supportsFocusDistance();
    case MediaConstraintType::WhiteBalanceMode:
        return supportsWhiteBalanceMode();
    case MediaConstraintType::Zoom:
        return supportsZoom();
    case MediaConstraintType::Torch:
        return supportsTorch();
    case MediaConstraintType::BackgroundBlur:
        return supportsBackgroundBlur();
    case MediaConstraintType::PowerEfficient:
        return false;
    }

    ASSERT_NOT_REACHED();
    return false;
}

} // namespace WebCore

#endif // ENABLE(MEDIA_STREAM)
