/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 5, 2022.
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
#include "Pattern.h"

#if USE(CAIRO)

#include "CairoUtilities.h"
#include "GraphicsContext.h"

namespace WebCore {

cairo_pattern_t* Pattern::createPlatformPattern(const AffineTransform&) const
{
    auto nativeImage = tileNativeImage();
    if (!nativeImage)
        return nullptr;

    auto platformImage = nativeImage->platformImage();
    if (!platformImage)
        return nullptr;

    cairo_pattern_t* pattern = cairo_pattern_create_for_surface(platformImage.get());

    // cairo merges patter space and user space itself
    cairo_matrix_t matrix = toCairoMatrix(patternSpaceTransform());
    cairo_matrix_invert(&matrix);
    cairo_pattern_set_matrix(pattern, &matrix);

    if (repeatX() || repeatY())
        cairo_pattern_set_extend(pattern, CAIRO_EXTEND_REPEAT);
    return pattern;
}

} // namespace WebCore

#endif // USE(CAIRO)
