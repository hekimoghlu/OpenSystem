/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 9, 2023.
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

#if PLATFORM(IOS) || PLATFORM(VISION)
#import <pal/system/ios/Device.h>
#endif

namespace WebCore {

enum class ScreenOrientationType : uint8_t {
    PortraitPrimary,
    PortraitSecondary,
    LandscapePrimary,
    LandscapeSecondary
};

constexpr bool isPortrait(ScreenOrientationType type)
{
    return type == ScreenOrientationType::PortraitPrimary || type == ScreenOrientationType::PortraitSecondary;
}

constexpr bool isLandscape(ScreenOrientationType type)
{
    return type == ScreenOrientationType::LandscapePrimary || type == ScreenOrientationType::LandscapeSecondary;
}

inline ScreenOrientationType naturalScreenOrientationType()
{
#if PLATFORM(IOS) || PLATFORM(VISION)
    if (PAL::deviceHasIPadCapability())
        return ScreenOrientationType::LandscapePrimary;
    return ScreenOrientationType::PortraitPrimary;
#elif PLATFORM(WATCHOS)
    return ScreenOrientationType::PortraitPrimary;
#else
    // On Desktop and TV, the natural orientation must be landscape-primary.
    return ScreenOrientationType::LandscapePrimary;
#endif
}

} // namespace WebCore
