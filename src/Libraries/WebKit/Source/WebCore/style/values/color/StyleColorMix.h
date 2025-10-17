/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 3, 2024.
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

#include "CSSColorMix.h"
#include "StyleColor.h"
#include "StylePrimitiveNumericTypes.h"
#include <optional>
#include <wtf/UniqueRef.h>

namespace WebCore {

class Color;

namespace Style {

struct ColorResolutionState;

struct ColorMix {
    WTF_MAKE_STRUCT_FAST_ALLOCATED;

    struct Component {
        using Percentage = Style::Percentage<CSS::Range{0, 100}>;

        Color color;
        std::optional<Percentage> percentage;

        bool operator==(const Component&) const = default;
    };

    ColorInterpolationMethod colorInterpolationMethod;
    Component mixComponents1;
    Component mixComponents2;

    bool operator==(const ColorMix&) const = default;
};

inline bool operator==(const UniqueRef<ColorMix>& a, const UniqueRef<ColorMix>& b)
{
    return a.get() == b.get();
}

Color toStyleColor(const CSS::ColorMix&, ColorResolutionState&);
WebCore::Color resolveColor(const ColorMix&, const WebCore::Color& currentColor);
bool containsCurrentColor(const ColorMix&);

void serializationForCSS(StringBuilder&, const ColorMix&);
String serializationForCSS(const ColorMix&);

WTF::TextStream& operator<<(WTF::TextStream&, const ColorMix&);

} // namespace Style
} // namespace WebCore
