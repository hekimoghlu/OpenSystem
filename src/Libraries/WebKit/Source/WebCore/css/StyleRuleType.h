/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 30, 2024.
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

// https://w3c.github.io/csswg-drafts/cssom-1/#the-cssrule-interface

enum class StyleRuleType : uint8_t {
    Style = 1,
    Charset = 2,
    Import = 3,
    Media = 4,
    FontFace = 5,
    Page = 6,
    Keyframes = 7,
    Keyframe = 8,
    Margin = 9,
    Namespace = 10,
    CounterStyle = 11,
    Supports = 12,
    FontFeatureValues = 14,
    // Numbers above 14 are not exposed to the web.
    ViewTransition = 15,
    LayerBlock,
    LayerStatement,
    Container,
    FontPaletteValues,
    FontFeatureValuesBlock,
    Property,
    StyleWithNesting,
    Scope,
    StartingStyle,
    NestedDeclarations,
    PositionTry
};

static constexpr auto firstUnexposedStyleRuleType = StyleRuleType::ViewTransition;

} // namespace WebCore
