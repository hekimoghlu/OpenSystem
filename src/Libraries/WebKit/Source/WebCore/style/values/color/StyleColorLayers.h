/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 20, 2024.
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

#include "CSSColorLayers.h"
#include "StyleColor.h"
#include <wtf/UniqueRef.h>

namespace WebCore {

enum class BlendMode : uint8_t;
class Color;

namespace Style {

struct ColorResolutionState;

struct ColorLayers {
    WTF_MAKE_STRUCT_FAST_ALLOCATED;

    BlendMode blendMode;
    CommaSeparatedVector<Color> colors;

    bool operator==(const ColorLayers&) const = default;
};

inline bool operator==(const UniqueRef<ColorLayers>& a, const UniqueRef<ColorLayers>& b)
{
    return a.get() == b.get();
}

Color toStyleColor(const CSS::ColorLayers&, ColorResolutionState&);
WebCore::Color resolveColor(const ColorLayers&, const WebCore::Color& currentColor);
bool containsCurrentColor(const ColorLayers&);

void serializationForCSS(StringBuilder&, const ColorLayers&);
String serializationForCSS(const ColorLayers&);

WTF::TextStream& operator<<(WTF::TextStream&, const ColorLayers&);

} // namespace Style
} // namespace WebCore
