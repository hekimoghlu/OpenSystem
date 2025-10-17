/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 10, 2023.
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

#include "DestinationColorSpace.h"
#include "FloatRect.h"
#include "PlatformScreen.h"
#include <wtf/HashMap.h>
#include <wtf/RetainPtr.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

struct ScreenData {
    FloatRect screenAvailableRect;
    FloatRect screenRect;
    DestinationColorSpace colorSpace { DestinationColorSpace::SRGB() };
    int screenDepth { 0 };
    int screenDepthPerComponent { 0 };
    bool screenSupportsExtendedColor { false };
    bool screenHasInvertedColors { false };
    bool screenSupportsHighDynamicRange { false };
#if PLATFORM(MAC)
    FloatSize screenSize; // In millimeters.
    bool screenIsMonochrome { false };
    uint32_t displayMask { 0 };
    PlatformGPUID gpuID { 0 };
    DynamicRangeMode preferredDynamicRangeMode { DynamicRangeMode::Standard };
    WEBCORE_EXPORT double screenDPI() const;
#endif
#if PLATFORM(GTK) || (PLATFORM(WPE) && ENABLE(WPE_PLATFORM))
    IntSize screenSize; // In millimeters.
    double dpi; // Already corrected for device scaling.
#endif

#if PLATFORM(MAC) || PLATFORM(IOS_FAMILY)
    float scaleFactor { 1 };
#endif
};

using ScreenDataMap = HashMap<PlatformDisplayID, ScreenData>;

struct ScreenProperties {
    PlatformDisplayID primaryDisplayID { 0 };
    ScreenDataMap screenDataMap;
};

} // namespace WebCore
