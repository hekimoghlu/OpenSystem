/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 11, 2023.
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

#include "CSSCalcSymbolsAllowed.h"
#include "CSSPrimitiveNumericRange.h"
#include "CSSPropertyParserOptions.h"
#include <optional>

namespace WebCore {

namespace Calculation {
enum class Category : uint8_t;
}

class CSSParserTokenRange;
struct CSSParserContext;

enum CSSValueID : uint16_t;

namespace CSSCalc {

struct SimplificationOptions;
struct Tree;

struct ParserOptions {
    // `category` represents the context in which the parse is taking place.
    Calculation::Category category;

    // `range` represents the allowed numeric range for the calculated result.
    CSS::Range range;

    // `allowedSymbols` contains additional symbols that can be used in the calculation. These will need to be resolved before the calculation can be resolved.
    CSSCalcSymbolsAllowed allowedSymbols;

    // `propertyOptions` contains options about the specific property the calc() is intended to be used with.
    CSSPropertyParserOptions propertyOptions;
};

// Parses and simplifies the provided `CSSParserTokenRange` into a CSSCalc::Tree. Returns `std::nullopt` on failure.
std::optional<Tree> parseAndSimplify(CSSParserTokenRange&, const CSSParserContext&, const ParserOptions&, const SimplificationOptions&);

// Returns whether the provided `CSSValueID` is one of the functions that should be parsed as a `calc()`.
bool isCalcFunction(CSSValueID function, const CSSParserContext&);

} // namespace CSSCalc
} // namespace WebCore
