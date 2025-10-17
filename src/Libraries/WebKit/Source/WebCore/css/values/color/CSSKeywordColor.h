/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 13, 2025.
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

#include "CSSValueTypes.h"
#include <wtf/Forward.h>

namespace WebCore {

class Color;

enum CSSValueID : uint16_t;
enum class StyleColorOptions : uint8_t;

namespace CSS {

struct PlatformColorResolutionState;

enum class ColorType : uint8_t;

struct KeywordColor {
    WTF_MAKE_STRUCT_FAST_ALLOCATED;

    CSSValueID valueID;

    bool operator==(const KeywordColor&) const = default;
};

WEBCORE_EXPORT bool isAbsoluteColorKeyword(CSSValueID);
WEBCORE_EXPORT bool isCurrentColorKeyword(CSSValueID);
WEBCORE_EXPORT bool isSystemColorKeyword(CSSValueID);
WEBCORE_EXPORT bool isDeprecatedSystemColorKeyword(CSSValueID);

bool isColorKeyword(CSSValueID);
bool isColorKeyword(CSSValueID, OptionSet<ColorType>);

WebCore::Color colorFromAbsoluteKeyword(CSSValueID);
WebCore::Color colorFromKeyword(CSSValueID, OptionSet<StyleColorOptions>);

WebCore::Color createColor(const KeywordColor&, PlatformColorResolutionState&);
bool containsCurrentColor(const KeywordColor&);
bool containsColorSchemeDependentColor(const KeywordColor&);

template<> struct Serialize<KeywordColor> { void operator()(StringBuilder&, const KeywordColor&); };
template<> struct ComputedStyleDependenciesCollector<KeywordColor> { constexpr void operator()(ComputedStyleDependencies&, const KeywordColor&) { } };
template<> struct CSSValueChildrenVisitor<KeywordColor> { constexpr IterationStatus operator()(const Function<IterationStatus(CSSValue&)>&, const KeywordColor&) { return IterationStatus::Continue; } };

} // namespace CSS
} // namespace WebCore
