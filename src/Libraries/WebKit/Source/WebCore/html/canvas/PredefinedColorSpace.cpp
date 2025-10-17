/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 30, 2022.
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
#include "PredefinedColorSpace.h"

#include "DestinationColorSpace.h"

namespace WebCore {

DestinationColorSpace toDestinationColorSpace(PredefinedColorSpace colorSpace)
{
    switch (colorSpace) {
    case PredefinedColorSpace::SRGB:
        return DestinationColorSpace::SRGB();
#if ENABLE(PREDEFINED_COLOR_SPACE_DISPLAY_P3)
    case PredefinedColorSpace::DisplayP3:
        return DestinationColorSpace::DisplayP3();
#endif
    }

    ASSERT_NOT_REACHED();
    return DestinationColorSpace::SRGB();
}

std::optional<PredefinedColorSpace> toPredefinedColorSpace(const DestinationColorSpace& colorSpace)
{
    if (colorSpace == DestinationColorSpace::SRGB())
        return PredefinedColorSpace::SRGB;
#if ENABLE(PREDEFINED_COLOR_SPACE_DISPLAY_P3)
    if (colorSpace == DestinationColorSpace::DisplayP3())
        return PredefinedColorSpace::DisplayP3;
#endif

    return std::nullopt;
}

}
