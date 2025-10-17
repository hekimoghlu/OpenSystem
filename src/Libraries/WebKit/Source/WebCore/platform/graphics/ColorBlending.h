/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 13, 2025.
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

namespace WebCore {

struct BlendingContext;
class Color;

// NOTE: These functions do a lossy conversion to 8-bit sRGBA before blending.

// This is an implementation of Porter-Duff's "source-over" equation.
WEBCORE_EXPORT Color blendSourceOver(const Color& backdrop, const Color& source);

// Bespoke "whitening" algorithm used by RenderTheme::transformSelectionBackgroundColor.
// Note: This is a no-op if the color to blend with isn't opaque, which is likely not what you were expecting.
Color blendWithWhite(const Color&);

Color blend(const Color& from, const Color& to, const BlendingContext&);
Color blendWithoutPremultiply(const Color& from, const Color& to, const BlendingContext&);

}
