/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 27, 2025.
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

#include "CSSPosition.h"
#include "CSSPropertyParserOptions.h"
#include <optional>

namespace WebCore {

class CSSParserTokenRange;
class CSSValue;

struct CSSParserContext;
enum class BoxOrient : bool;

namespace CSSPropertyParserHelpers {

// MARK: <position> | <bg-position>
// https://drafts.csswg.org/css-values/#position

enum class PositionSyntax {
    Position, // <position>
    BackgroundPosition // <bg-position>
};

struct PositionCoordinates {
    Ref<CSSValue> x;
    Ref<CSSValue> y;
};

RefPtr<CSSValue> consumePosition(CSSParserTokenRange&, const CSSParserContext&, UnitlessQuirk, PositionSyntax);

RefPtr<CSSValue> consumePositionX(CSSParserTokenRange&, const CSSParserContext&);
RefPtr<CSSValue> consumePositionY(CSSParserTokenRange&, const CSSParserContext&);

std::optional<PositionCoordinates> consumePositionCoordinates(CSSParserTokenRange&, const CSSParserContext&, UnitlessQuirk, PositionSyntax);
std::optional<PositionCoordinates> consumeOneOrTwoValuedPositionCoordinates(CSSParserTokenRange&, const CSSParserContext&, UnitlessQuirk);

// MARK: <position> (unresolved)
std::optional<CSS::Position> consumePositionUnresolved(CSSParserTokenRange&, const CSSParserContext&);
std::optional<CSS::Position> consumeOneOrTwoComponentPositionUnresolved(CSSParserTokenRange&, const CSSParserContext&);

std::optional<CSS::TwoComponentPositionHorizontal> consumeTwoComponentPositionHorizontalUnresolved(CSSParserTokenRange&, const CSSParserContext&);
std::optional<CSS::TwoComponentPositionVertical> consumeTwoComponentPositionVerticalUnresolved(CSSParserTokenRange&, const CSSParserContext&);

} // namespace CSSPropertyParserHelpers
} // namespace WebCore
