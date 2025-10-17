/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 26, 2022.
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

#include "Color.h"
#include "FloatSize.h"
#include <wtf/Vector.h>

namespace WTF {
class TextStream;
}

namespace WebCore {

// Legacy shadow blur radius is used for canvas, and -webkit-box-shadow.
// It has different treatment of radii > 8px.
enum class ShadowRadiusMode : bool {
    Default,
    Legacy
};

struct GraphicsDropShadow {
    FloatSize offset;
    float radius;
    Color color;
    ShadowRadiusMode radiusMode { ShadowRadiusMode::Default };
    float opacity { 1 };

    bool isVisible() const { return color.isVisible(); }
    bool isBlurred() const { return isVisible() && radius; }
    bool hasOutsets() const { return isBlurred() || (isVisible() && !offset.isZero()); }
};

inline bool operator==(const GraphicsDropShadow& a, const GraphicsDropShadow& b)
{
    return a.offset == b.offset && a.radius == b.radius && a.color == b.color;
}

struct GraphicsGaussianBlur {
    FloatSize radius;

    friend bool operator==(const GraphicsGaussianBlur&, const GraphicsGaussianBlur&) = default;
};

struct GraphicsColorMatrix {
    std::array<float, 20> values;

    friend bool operator==(const GraphicsColorMatrix&, const GraphicsColorMatrix&) = default;
};

using GraphicsStyle = std::variant<
    GraphicsDropShadow,
    GraphicsGaussianBlur,
    GraphicsColorMatrix
>;

WEBCORE_EXPORT WTF::TextStream& operator<<(WTF::TextStream&, const GraphicsDropShadow&);
WEBCORE_EXPORT WTF::TextStream& operator<<(WTF::TextStream&, const GraphicsGaussianBlur&);
WEBCORE_EXPORT WTF::TextStream& operator<<(WTF::TextStream&, const GraphicsColorMatrix&);
WEBCORE_EXPORT WTF::TextStream& operator<<(WTF::TextStream&, const GraphicsStyle&);

} // namespace WebCore
