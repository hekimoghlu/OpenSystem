/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 22, 2025.
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
#include "CSSContrastColorResolver.h"

#include "ColorLuminance.h"

namespace WebCore {
namespace CSS {

// https://drafts.csswg.org/css-color-5/#contrast-color
WebCore::Color resolve(const ContrastColorResolver& resolver)
{
    // FIXME: Implement support for a non-naive resolution.

    auto luminance = relativeLuminance(resolver.color);

    auto contrastWithWhite = contrastRatio(1.0, luminance);
    auto contrastWithBlack = contrastRatio(0.0, luminance);

    return contrastWithWhite > contrastWithBlack ? WebCore::Color::white : WebCore::Color::black;
}

} // namespace CSS
} // namespace WebCore
