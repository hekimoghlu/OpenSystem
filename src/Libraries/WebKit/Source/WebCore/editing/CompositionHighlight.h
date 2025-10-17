/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 4, 2024.
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

#include "Color.h"

namespace WebCore {

struct CompositionHighlight {
    CompositionHighlight() = default;
    CompositionHighlight(unsigned startOffset, unsigned endOffset, const std::optional<Color>& backgroundColor, const std::optional<Color>& foregroundColor)
        : startOffset(startOffset)
        , endOffset(endOffset)
        , backgroundColor(backgroundColor)
        , foregroundColor(foregroundColor)
    {
    }

#if PLATFORM(IOS_FAMILY)
    static constexpr auto defaultCompositionFillColor = SRGBA<uint8_t> { 175, 192, 227, 60 };
#else
    static constexpr auto defaultCompositionFillColor = SRGBA<uint8_t> { 225, 221, 85 };
#endif

    unsigned startOffset { 0 };
    unsigned endOffset { 0 };
    std::optional<Color> backgroundColor;
    std::optional<Color> foregroundColor;
};

} // namespace WebCore
