/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 7, 2024.
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
#include "ColorUtilities.h"

namespace WebCore {

SRGBA<float> premultiplied(const SRGBA<float>& color)
{
    auto [r, g, b, a] = color.resolved();
    return { r * a, g * a, b * a, a };
}

SRGBA<float> unpremultiplied(const SRGBA<float>& color)
{
    auto [r, g, b, a] = color.resolved();
    if (!a)
        return color;
    return makeFromComponentsClampingExceptAlpha<SRGBA<float>>(r / a, g / a, b / a, a);
}

SRGBA<uint8_t> premultipliedFlooring(SRGBA<uint8_t> color)
{
    auto [r, g, b, a] = color.resolved();
    if (!a)
        return { 0, 0, 0, 0 };
    if (a == 255)
        return color;
    return makeFromComponentsClampingExceptAlpha<SRGBA<uint8_t>>(fastDivideBy255(r * a), fastDivideBy255(g * a), fastDivideBy255(b * a), a);
}

SRGBA<uint8_t> premultipliedCeiling(SRGBA<uint8_t> color)
{
    auto [r, g, b, a] = color.resolved();
    if (!a)
        return { 0, 0, 0, 0 };
    if (a == 255)
        return color;
    return makeFromComponentsClampingExceptAlpha<SRGBA<uint8_t>>(fastDivideBy255(r * a + 254), fastDivideBy255(g * a + 254), fastDivideBy255(b * a + 254), a);
}

static inline uint16_t unpremultipliedComponentByte(uint8_t c, uint8_t a)
{
    return (fastMultiplyBy255(c) + a - 1) / a;
}

SRGBA<uint8_t> unpremultiplied(SRGBA<uint8_t> color)
{
    auto [r, g, b, a] = color.resolved();
    if (!a || a == 255)
        return color;
    return makeFromComponentsClampingExceptAlpha<SRGBA<uint8_t>>(unpremultipliedComponentByte(r, a), unpremultipliedComponentByte(g, a), unpremultipliedComponentByte(b, a), a);
}

} // namespace WebCore
