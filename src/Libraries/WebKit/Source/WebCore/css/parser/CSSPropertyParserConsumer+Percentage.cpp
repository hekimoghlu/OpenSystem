/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 27, 2023.
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
#include "CSSPropertyParserConsumer+Percentage.h"
#include "CSSPropertyParserConsumer+PercentageDefinitions.h"

#include "CSSCalcSymbolTable.h"
#include "CSSParserContext.h"
#include "CSSPropertyParserConsumer+CSSPrimitiveValueResolver.h"
#include "CSSPropertyParserConsumer+NumberDefinitions.h"

namespace WebCore {
namespace CSSPropertyParserHelpers {

// MARK: - Consumer functions

RefPtr<CSSPrimitiveValue> consumePercentage(CSSParserTokenRange& range, const CSSParserContext& context, ValueRange valueRange)
{
    if (valueRange == ValueRange::All)
        return CSSPrimitiveValueResolver<CSS::Percentage<CSS::All>>::consumeAndResolve(range, context, { });
    return CSSPrimitiveValueResolver<CSS::Percentage<CSS::Nonnegative>>::consumeAndResolve(range, context, { });
}

template<auto R> static RefPtr<CSSPrimitiveValue> consumePercentageDividedBy100OrNumber(CSSParserTokenRange& range, const CSSParserContext& context)
{
    using NumberConsumer = ConsumerDefinition<CSS::Number<R>>;
    using PercentageConsumer = ConsumerDefinition<CSS::Percentage<R>>;

    auto& token = range.peek();

    switch (token.type()) {
    case FunctionToken:
        if (auto value = NumberConsumer::FunctionToken::consume(range, context, { }, { }))
            return CSSPrimitiveValueResolver<CSS::Number<R>>::resolve(*value, { });
        if (auto value = PercentageConsumer::FunctionToken::consume(range, context, { }, { }))
            return CSSPrimitiveValueResolver<CSS::Percentage<R>>::resolve(*value, { });
        break;

    case NumberToken:
        if (auto value = NumberConsumer::NumberToken::consume(range, context, { }, { }))
            return CSSPrimitiveValueResolver<CSS::Number<R>>::resolve(*value, { });
        break;

    case PercentageToken:
        if (auto value = PercentageConsumer::PercentageToken::consume(range, context, { }, { }))
            return CSSPrimitiveValue::create(value->value / 100.0);
        break;

    default:
        break;
    }

    return nullptr;
}

RefPtr<CSSPrimitiveValue> consumePercentageDividedBy100OrNumber(CSSParserTokenRange& range, const CSSParserContext& context, ValueRange valueRange)
{
    if (valueRange == ValueRange::All)
        return consumePercentageDividedBy100OrNumber<CSS::All>(range, context);
    return consumePercentageDividedBy100OrNumber<CSS::Nonnegative>(range, context);
}

} // namespace CSSPropertyParserHelpers
} // namespace WebCore
