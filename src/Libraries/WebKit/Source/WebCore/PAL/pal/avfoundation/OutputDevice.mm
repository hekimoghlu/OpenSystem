/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 11, 2025.
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
#include "OutputDevice.h"

#if USE(AVFOUNDATION)

#include <pal/spi/cocoa/AVFoundationSPI.h>

#include <pal/cocoa/AVFoundationSoftLink.h>

// FIXME(rdar://70358894): Remove once -allowsHeadTrackedSpatialAudio lands:
@interface AVOutputDevice (AllowsHeadTrackedSpatialAudio)
- (BOOL)allowsHeadTrackedSpatialAudio;
@end

namespace PAL {

OutputDevice::OutputDevice(RetainPtr<AVOutputDevice>&& device)
    : m_device(WTFMove(device))
{
}

String OutputDevice::name() const
{
ALLOW_DEPRECATED_DECLARATIONS_BEGIN
    return [m_device name];
ALLOW_DEPRECATED_DECLARATIONS_END
}

uint8_t OutputDevice::deviceFeatures() const
{
    auto avDeviceFeatures = [m_device deviceFeatures];
    uint8_t deviceFeatures { 0 };
    if (avDeviceFeatures & AVOutputDeviceFeatureAudio)
        deviceFeatures |= (uint8_t)DeviceFeatures::Audio;
    if (avDeviceFeatures & AVOutputDeviceFeatureScreen)
        deviceFeatures |= (uint8_t)DeviceFeatures::Screen;
    if (avDeviceFeatures & AVOutputDeviceFeatureVideo)
        deviceFeatures |= (uint8_t)DeviceFeatures::Video;
    return deviceFeatures;
}

bool OutputDevice::supportsSpatialAudio() const
{
    if (![m_device respondsToSelector:@selector(supportsHeadTrackedSpatialAudio)]
        || ![m_device supportsHeadTrackedSpatialAudio])
        return false;

    return ![m_device respondsToSelector:@selector(allowsHeadTrackedSpatialAudio)]
        || [m_device allowsHeadTrackedSpatialAudio];
}

}

#endif
