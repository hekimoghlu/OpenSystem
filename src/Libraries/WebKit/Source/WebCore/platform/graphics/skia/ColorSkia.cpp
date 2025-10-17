/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 14, 2023.
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

#if USE(SKIA)

namespace WebCore {

Color::Color(const SkColor& skColor)
    : Color(SRGBA<uint8_t> { static_cast<uint8_t>(SkColorGetR(skColor)), static_cast<uint8_t>(SkColorGetG(skColor)), static_cast<uint8_t>(SkColorGetB(skColor)), static_cast<uint8_t>(SkColorGetA(skColor)) })
{
}

Color::operator SkColor() const
{
    auto [r, g, b, a] = toColorTypeLossy<SRGBA<uint8_t>>().resolved();
    return SkColorSetARGB(a, r, g, b);
}

Color::Color(const SkColor4f& skColor)
    : Color(convertColor<SRGBA<uint8_t>>(SRGBA<float> { skColor.fR, skColor.fG, skColor.fB, skColor.fA }))
{
}

Color::operator SkColor4f() const
{
    auto [r, g, b, a] = toColorTypeLossy<SRGBA<float>>().resolved();
    return { r, g, b, a };
}

} // namespace WebCore

#endif // USE(SKIA)

