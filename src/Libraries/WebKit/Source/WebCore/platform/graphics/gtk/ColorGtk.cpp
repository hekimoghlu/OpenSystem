/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 23, 2024.
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
#include "Color.h"

#include <gdk/gdk.h>

namespace WebCore {

Color::Color(const GdkRGBA& color)
    : Color(convertColor<SRGBA<uint8_t>>(SRGBA<float> { static_cast<float>(color.red), static_cast<float>(color.green), static_cast<float>(color.blue), static_cast<float>(color.alpha) }))
{
}

Color::operator GdkRGBA() const
{
    auto [r, g, b, a] = toColorTypeLossy<SRGBA<float>>().resolved();
    return { r, g, b, a };
}

}

// vim: ts=4 sw=4 et
