/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 4, 2025.
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

#include "CSSResolvedColor.h"
#include "Color.h"
#include <wtf/Forward.h>

namespace WebCore {
namespace Style {

struct Color;
struct ColorResolutionState;

struct ResolvedColor {
    WTF_MAKE_STRUCT_FAST_ALLOCATED;

    WebCore::Color color;

    bool operator==(const ResolvedColor&) const = default;
};

Color toStyleColor(const CSS::ResolvedColor&, ColorResolutionState&);

inline WebCore::Color resolveColor(const ResolvedColor& absoluteColor, const WebCore::Color&)
{
    return absoluteColor.color;
}

constexpr bool containsCurrentColor(const ResolvedColor&)
{
    return false;
}

void serializationForCSS(StringBuilder&, const ResolvedColor&);
String serializationForCSS(const ResolvedColor&);

WTF::TextStream& operator<<(WTF::TextStream&, const ResolvedColor&);

} // namespace Style
} // namespace WebCore
