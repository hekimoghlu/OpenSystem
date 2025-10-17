/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 23, 2025.
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
#include "ColorLuminance.h"

#include "Color.h"

namespace WebCore {

double relativeLuminance(const Color& color)
{
    return color.callOnUnderlyingType([&] (const auto& underlyingColor) {
        return relativeLuminance(underlyingColor);
    });
}

double contrastRatio(const Color& colorA, const Color& colorB)
{
    return colorA.callOnUnderlyingType([&] (const auto& underlyingColorA) {
        return colorB.callOnUnderlyingType([&] (const auto& underlyingColorB) {
            return contrastRatio(underlyingColorA, underlyingColorB);
        });
    });
}

}
