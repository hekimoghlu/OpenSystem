/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 28, 2023.
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

#include "CSSPropertyParserConsumer+MetaConsumerDefinitions.h"

namespace WebCore {
namespace CSSPropertyParserHelpers {

struct IntegerValidator {
    static constexpr std::optional<CSS::IntegerUnit> validate(CSSUnitType unitType, CSSPropertyParserOptions)
    {
        return CSS::UnitTraits<CSS::IntegerUnit>::validate(unitType);
    }

    template<auto R, typename V> static bool isValid(CSS::IntegerRaw<R, V> raw, CSSPropertyParserOptions)
    {
        return isValidCanonicalValue(raw);
    }
};

template<typename Primitive, typename Validator> struct NumberConsumerForIntegerValues {
    static constexpr CSSParserTokenType tokenType = NumberToken;

    static std::optional<typename Primitive::Raw> consume(CSSParserTokenRange& range, const CSSParserContext&, CSSCalcSymbolsAllowed, CSSPropertyParserOptions options)
    {
        ASSERT(range.peek().type() == NumberToken);

        if (range.peek().numericValueType() != IntegerValueType)
            return std::nullopt;

        auto rawValue = typename Primitive::Raw { CSS::IntegerUnit::Integer, range.peek().numericValue() };

        if constexpr (rawValue.range.options != CSS::RangeOptions::Default)
            rawValue = performParseTimeClamp(rawValue);

        if (!Validator::isValid(rawValue, options))
            return std::nullopt;

        range.consumeIncludingWhitespace();
        return rawValue;
    }
};

template<CSS::Range R, typename IntType>
struct ConsumerDefinition<CSS::Integer<R, IntType>> {
    using FunctionToken = FunctionConsumerForCalcValues<CSS::Integer<R, IntType>>;
    using NumberToken = NumberConsumerForIntegerValues<CSS::Integer<R, IntType>, IntegerValidator>;
};

} // namespace CSSPropertyParserHelpers
} // namespace WebCore
