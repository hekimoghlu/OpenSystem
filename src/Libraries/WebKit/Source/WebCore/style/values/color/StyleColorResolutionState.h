/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 22, 2024.
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

#include "CSSToLengthConversionData.h"

namespace WebCore {

class Document;
class RenderStyle;

namespace Style {

enum class ForVisitedLink : bool;

// Used to resolve a CSS::Color to a Style::Color during style building.

struct ColorResolutionState {
    Ref<const Document> document;
    const RenderStyle& style;
    CSSToLengthConversionData conversionData;
    ForVisitedLink forVisitedLink;

    unsigned nestingLevel = 0;
};

// RAII helper to increment/decrement nesting level.
struct ColorResolutionStateNester {
    ColorResolutionStateNester(ColorResolutionState& state)
        : state { state }
    {
        state.nestingLevel++;
    }

    ~ColorResolutionStateNester()
    {
        state.nestingLevel--;
    }

    ColorResolutionState& state;
};

} // namespace Style
} // namespace WebCore
