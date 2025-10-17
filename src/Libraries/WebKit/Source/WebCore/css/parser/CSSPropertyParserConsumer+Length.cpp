/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 15, 2021.
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
#include "config.h"
#include "CSSPropertyParserConsumer+Length.h"
#include "CSSPropertyParserConsumer+LengthDefinitions.h"

#include "CSSCalcSymbolTable.h"
#include "CSSParserContext.h"
#include "CSSPropertyParserConsumer+CSSPrimitiveValueResolver.h"

namespace WebCore {
namespace CSSPropertyParserHelpers {

RefPtr<CSSPrimitiveValue> consumeLength(CSSParserTokenRange& range, const CSSParserContext& context, ValueRange valueRange, UnitlessQuirk unitless)
{
    const auto options = CSSPropertyParserOptions {
        .parserMode = context.mode,
        .unitless = unitless,
        .unitlessZero = UnitlessZeroQuirk::Allow
    };
    if (valueRange == ValueRange::All)
        return CSSPrimitiveValueResolver<CSS::Length<CSS::All>>::consumeAndResolve(range, context, options);
    return CSSPrimitiveValueResolver<CSS::Length<CSS::Nonnegative>>::consumeAndResolve(range, context, options);
}

RefPtr<CSSPrimitiveValue> consumeLength(CSSParserTokenRange& range, const CSSParserContext& context, CSSParserMode overrideParserMode, ValueRange valueRange, UnitlessQuirk unitless)
{
    const auto options = CSSPropertyParserOptions {
        .parserMode = overrideParserMode,
        .unitless = unitless,
        .unitlessZero = UnitlessZeroQuirk::Allow
    };
    if (valueRange == ValueRange::All)
        return CSSPrimitiveValueResolver<CSS::Length<CSS::All>>::consumeAndResolve(range, context, options);
    return CSSPrimitiveValueResolver<CSS::Length<CSS::Nonnegative>>::consumeAndResolve(range, context, options);
}

} // namespace CSSPropertyParserHelpers
} // namespace WebCore
