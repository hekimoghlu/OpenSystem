/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 22, 2023.
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

#include "MediaConstraintType.h"

namespace WebCore {

class RealtimeMediaSourceSupportedConstraints {
public:
    RealtimeMediaSourceSupportedConstraints()
    {
    }
    
    RealtimeMediaSourceSupportedConstraints(bool supportsWidth, bool supportsHeight, bool supportsAspectRatio, bool supportsFrameRate, bool supportsFacingMode, bool supportsVolume, bool supportsSampleRate, bool supportsSampleSize, bool supportsEchoCancellation, bool supportsDeviceId, bool supportsGroupId, bool supportsDisplaySurface, bool supportsLogicalSurface, bool supportsFocusDistance, bool supportsWhiteBalanceMode, bool supportsZoom, bool supportsTorch, bool supportsBackgroundBlur, bool supportsPowerEfficient)
        : m_supportsWidth(supportsWidth)
        , m_supportsHeight(supportsHeight)
        , m_supportsAspectRatio(supportsAspectRatio)
        , m_supportsFrameRate(supportsFrameRate)
        , m_supportsFacingMode(supportsFacingMode)
        , m_supportsVolume(supportsVolume)
        , m_supportsSampleRate(supportsSampleRate)
        , m_supportsSampleSize(supportsSampleSize)
        , m_supportsEchoCancellation(supportsEchoCancellation)
        , m_supportsDeviceId(supportsDeviceId)
        , m_supportsGroupId(supportsGroupId)
        , m_supportsDisplaySurface(supportsDisplaySurface)
        , m_supportsLogicalSurface(supportsLogicalSurface)
        , m_supportsFocusDistance(supportsFocusDistance)
        , m_supportsWhiteBalanceMode(supportsWhiteBalanceMode)
        , m_supportsZoom(supportsZoom)
        , m_supportsTorch(supportsTorch)
        , m_supportsBackgroundBlur(supportsBackgroundBlur)
        , m_supportsPowerEfficient(supportsPowerEfficient)
    {
    }

    bool supportsWidth() const { return m_supportsWidth; }
    void setSupportsWidth(bool value) { m_supportsWidth = value; }

    bool supportsHeight() const { return m_supportsHeight; }
    void setSupportsHeight(bool value) { m_supportsHeight = value; }

    bool supportsAspectRatio() const { return m_supportsAspectRatio; }
    void setSupportsAspectRatio(bool value) { m_supportsAspectRatio = value; }

    bool supportsFrameRate() const { return m_supportsFrameRate; }
    void setSupportsFrameRate(bool value) { m_supportsFrameRate = value; }

    bool supportsFacingMode() const { return m_supportsFacingMode; }
    void setSupportsFacingMode(bool value) { m_supportsFacingMode = value; }

    bool supportsVolume() const { return m_supportsVolume; }
    void setSupportsVolume(bool value) { m_supportsVolume = value; }

    bool supportsSampleRate() const { return m_supportsSampleRate; }
    void setSupportsSampleRate(bool value) { m_supportsSampleRate = value; }

    bool supportsSampleSize() const { return m_supportsSampleSize; }
    void setSupportsSampleSize(bool value) { m_supportsSampleSize = value; }

    bool supportsEchoCancellation() const { return m_supportsEchoCancellation; }
    void setSupportsEchoCancellation(bool value) { m_supportsEchoCancellation = value; }

    bool supportsDeviceId() const { return m_supportsDeviceId; }
    void setSupportsDeviceId(bool value) { m_supportsDeviceId = value; }

    bool supportsGroupId() const { return m_supportsGroupId; }
    void setSupportsGroupId(bool value) { m_supportsGroupId = value; }

    bool supportsDisplaySurface() const { return m_supportsDisplaySurface; }
    void setSupportsDisplaySurface(bool value) { m_supportsDisplaySurface = value; }

    bool supportsLogicalSurface() const { return m_supportsLogicalSurface; }
    void setSupportsLogicalSurface(bool value) { m_supportsLogicalSurface = value; }

    bool supportsConstraint(MediaConstraintType) const;

    bool supportsFocusDistance() const { return m_supportsFocusDistance; }
    void setSupportsFocusDistance(bool value) { m_supportsFocusDistance = value; }

    bool supportsWhiteBalanceMode() const { return m_supportsWhiteBalanceMode; }
    void setSupportsWhiteBalanceMode(bool value) { m_supportsWhiteBalanceMode = value; }

    bool supportsZoom() const { return m_supportsZoom; }
    void setSupportsZoom(bool value) { m_supportsZoom = value; }

    bool supportsTorch() const { return m_supportsTorch; }
    void setSupportsTorch(bool value) { m_supportsTorch = value; }

    bool supportsBackgroundBlur() const { return m_supportsBackgroundBlur; }
    void setSupportsBackgroundBlur(bool value) { m_supportsBackgroundBlur = value; }

    bool supportsPowerEfficient() const { return m_supportsPowerEfficient; }
    void setSupportsPowerEfficient(bool value) { m_supportsPowerEfficient = value; }

private:
    bool m_supportsWidth { false };
    bool m_supportsHeight { false };
    bool m_supportsAspectRatio { false };
    bool m_supportsFrameRate { false };
    bool m_supportsFacingMode { false };
    bool m_supportsVolume { false };
    bool m_supportsSampleRate { false };
    bool m_supportsSampleSize { false };
    bool m_supportsEchoCancellation { false };
    bool m_supportsDeviceId { false };
    bool m_supportsGroupId { false };
    bool m_supportsDisplaySurface { false };
    bool m_supportsLogicalSurface { false };
    bool m_supportsFocusDistance { false };
    bool m_supportsWhiteBalanceMode { false };
    bool m_supportsZoom { false };
    bool m_supportsTorch { false };
    bool m_supportsBackgroundBlur { false };
    bool m_supportsPowerEfficient { false };
};

} // namespace WebCore

#endif // ENABLE(MEDIA_STREAM)
