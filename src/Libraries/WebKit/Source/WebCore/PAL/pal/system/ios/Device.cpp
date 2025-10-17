/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 30, 2024.
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
#include "Device.h"

#if PLATFORM(IOS_FAMILY)

#include <mutex>
#include <pal/spi/ios/MobileGestaltSPI.h>
#include <wtf/NeverDestroyed.h>
#include <wtf/RetainPtr.h>
#include <wtf/text/WTFString.h>

namespace PAL {

bool deviceClassIsSmallScreen()
{
#if ENABLE(FORCE_DEVICE_CLASS_SMALL_SCREEN)
    return true;
#else
    static auto deviceClass = MGGetSInt32Answer(kMGQDeviceClassNumber, MGDeviceClassInvalid);
    return deviceClass == MGDeviceClassiPhone || deviceClass == MGDeviceClassiPod || deviceClass == MGDeviceClassWatch;
#endif
}

bool deviceClassIsVision()
{
#if PLATFORM(VISION)
    static auto deviceClass = MGGetSInt32Answer(kMGQDeviceClassNumber, MGDeviceClassInvalid);
    return deviceClass == MGDeviceClassRealityDevice;
#else
    return false;
#endif
}

String deviceName()
{
#if ENABLE(MOBILE_GESTALT_DEVICE_NAME)
    static NeverDestroyed<RetainPtr<CFStringRef>> deviceName;
    static std::once_flag onceKey;
    std::call_once(onceKey, [] {
        deviceName.get() = adoptCF(static_cast<CFStringRef>(MGCopyAnswer(kMGQDeviceName, nullptr)));
    });
    return deviceName.get().get();
#else
    if (!deviceClassIsSmallScreen())
        return "iPad"_s;
    return "iPhone"_s;
#endif
}

bool deviceHasIPadCapability()
{
    static bool deviceHasIPadCapability = MGGetBoolAnswer(kMGQiPadCapability);
    return deviceHasIPadCapability;
}

} // namespace PAL

#endif // PLATFORM(IOS_FAMILY)
