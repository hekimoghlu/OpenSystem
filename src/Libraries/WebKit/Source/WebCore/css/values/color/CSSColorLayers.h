/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 1, 2022.
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

#include "CSSColor.h"
#include <wtf/Vector.h>

namespace WebCore {

class Color;
enum class BlendMode : uint8_t;

namespace CSS {

struct PlatformColorResolutionState;

struct ColorLayers {
    WTF_MAKE_STRUCT_FAST_ALLOCATED;

    BlendMode blendMode;
    CommaSeparatedVector<Color> colors;

    bool operator==(const ColorLayers&) const = default;
};


WebCore::Color createColor(const ColorLayers&, PlatformColorResolutionState&);
bool containsCurrentColor(const ColorLayers&);
bool containsColorSchemeDependentColor(const ColorLayers&);

template<> struct Serialize<ColorLayers> { void operator()(StringBuilder&, const ColorLayers&); };
template<> struct ComputedStyleDependenciesCollector<ColorLayers> { void operator()(ComputedStyleDependencies&, const ColorLayers&); };
template<> struct CSSValueChildrenVisitor<ColorLayers> { IterationStatus operator()(const Function<IterationStatus(CSSValue&)>&, const ColorLayers&); };

} // namespace CSS
} // namespace WebCore
