/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 12, 2022.
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
#include "CSSCalcValue.h"
#include "CSSParserTokenRange.h"
#include "CSSPrimitiveNumericTypes.h"
#include "CSSPropertyParserOptions.h"

namespace WebCore {

struct CSSParserContext;

namespace CSSPropertyParserHelpers {

// MARK: - Generic Consumer Definition

template<typename> struct ConsumerDefinition;

inline bool shouldAcceptUnitlessValue(double value, CSSPropertyParserOptions options)
{
    // FIXME: Presentational HTML attributes shouldn't use the CSS parser for lengths.

    if (!value && options.unitlessZero == UnitlessZeroQuirk::Allow)
        return true;

    if (isUnitlessValueParsingEnabledForMode(options.parserMode))
        return true;

    return options.parserMode == HTMLQuirksMode && options.unitless == UnitlessQuirk::Allow;
}

// FIXME: Bailing on infinity during validation does not seem to match the intent of the spec,
// though due to the use of "implementation-defined" it may still be conforming. The spec states:
//
//   "When a value cannot be explicitly supported due to range/precision limitations, it must
//    be converted to the closest value supported by the implementation, but how the implementation
//    defines "closest" is implementation-defined as well."
//
// Angles have the additional restriction that:
//
//   "If an <angle> must be converted due to exceeding the implementation-defined range of supported
//    values, it must be clamped to the nearest supported multiple of 360deg."
//
// (https://drafts.csswg.org/css-values-4/#numeric-types)
//
// The infinity here is produced by the parser when a parsed number is no representable in
// as a double. A potentially more appropriate behavior would be to have the parser use
// std::numeric_limits<double>::max() instead. For angles, this would require further integration
// with the fast_float library (or whatever is currently being used to parse the number) to
// extract the correct modulo 360deg value.

// Shared validator for types dimensional types that need to canonicalize to support range
// constraints other than 0 and +/-âˆž.
template<typename Raw, typename F> bool isValidDimensionValue(Raw raw, F&& functor)
{
    if (std::isinf(raw.value))
        return false;

    if constexpr (raw.range.min == -CSS::Range::infinity && raw.range.max == CSS::Range::infinity)
        return true;
    else if constexpr (raw.range.min == 0 && raw.range.max == CSS::Range::infinity)
        return raw.value >= 0;
    else if constexpr (raw.range.min == -CSS::Range::infinity && raw.range.max == 0)
        return raw.value <= 0;
    else
        return functor();
}

// Shared validator for types that only support 0 and +/-âˆž as valid range constraints.
template<typename Raw> bool isValidNonCanonicalizableDimensionValue(Raw raw)
{
    if (std::isinf(raw.value))
        return false;

    if constexpr (raw.range.min == -CSS::Range::infinity && raw.range.max == CSS::Range::infinity)
        return true;
    else if constexpr (raw.range.min == 0 && raw.range.max == CSS::Range::infinity)
        return raw.value >= 0;
    else if constexpr (raw.range.min == -CSS::Range::infinity && raw.range.max == 0)
        return raw.value <= 0;
}

// Shared validator for types that always have their value in canonical units (number, percentage, flex).
template<typename Raw> bool isValidCanonicalValue(Raw raw)
{
    if (std::isinf(raw.value))
        return false;

    if constexpr (raw.range.min == -CSS::Range::infinity && raw.range.max == CSS::Range::infinity)
        return true;
    else if constexpr (raw.range.max == CSS::Range::infinity)
        return raw.value >= raw.range.min;
    else if constexpr (raw.range.min == -CSS::Range::infinity)
        return raw.value <= raw.range.max;
    else
        return raw.value >= raw.range.min && raw.value <= raw.range.max;
}

// Shared clamping utility.
template<typename Raw> Raw performParseTimeClamp(Raw raw)
{
    static_assert(raw.range.options != CSS::RangeOptions::Default);

    if constexpr (raw.range.options == CSS::RangeOptions::ClampLower)
        return { std::max<typename Raw::ResolvedValueType>(raw.value, raw.range.min) };
    else if constexpr (raw.range.options == CSS::RangeOptions::ClampUpper)
        return { std::min<typename Raw::ResolvedValueType>(raw.value, raw.range.max) };
    else if constexpr (raw.range.options == CSS::RangeOptions::ClampBoth)
        return { std::clamp<typename Raw::ResolvedValueType>(raw.value, raw.range.min, raw.range.max) };
}

// Shared consumer for `Dimension` tokens.
template<typename Primitive, typename Validator> struct DimensionConsumer {
    static constexpr CSSParserTokenType tokenType = DimensionToken;

    static std::optional<typename Primitive::Raw> consume(CSSParserTokenRange& range, const CSSParserContext&, CSSCalcSymbolsAllowed, CSSPropertyParserOptions options)
    {
        ASSERT(range.peek().type() == DimensionToken);

        auto& token = range.peek();

        auto validatedUnit = Validator::validate(token.unitType(), options);
        if (!validatedUnit)
            return std::nullopt;

        auto rawValue = typename Primitive::Raw { *validatedUnit, token.numericValue() };

        if constexpr (rawValue.range.options != CSS::RangeOptions::Default)
            rawValue = performParseTimeClamp(rawValue);

        if (!Validator::isValid(rawValue, options))
            return std::nullopt;

        range.consumeIncludingWhitespace();
        return rawValue;
    }
};

// Shared consumer for `Percentage` tokens.
template<typename Primitive, typename Validator> struct PercentageConsumer {
    static constexpr CSSParserTokenType tokenType = PercentageToken;

    static std::optional<typename Primitive::Raw> consume(CSSParserTokenRange& range, const CSSParserContext&, CSSCalcSymbolsAllowed, CSSPropertyParserOptions options)
    {
        ASSERT(range.peek().type() == PercentageToken);

        auto rawValue = typename Primitive::Raw { CSS::PercentageUnit::Percentage, range.peek().numericValue() };

        if constexpr (rawValue.range.options != CSS::RangeOptions::Default)
            rawValue = performParseTimeClamp(rawValue);

        if (!Validator::isValid(rawValue, options))
            return std::nullopt;

        range.consumeIncludingWhitespace();
        return rawValue;
    }
};

// Shared consumer for `Number` tokens.
template<typename Primitive, typename Validator> struct NumberConsumer {
    static constexpr CSSParserTokenType tokenType = NumberToken;

    static std::optional<typename Primitive::Raw> consume(CSSParserTokenRange& range, const CSSParserContext&, CSSCalcSymbolsAllowed, CSSPropertyParserOptions options)
    {
        ASSERT(range.peek().type() == NumberToken);

        auto rawValue = typename Primitive::Raw { CSS::NumberUnit::Number, range.peek().numericValue() };

        if constexpr (rawValue.range.options != CSS::RangeOptions::Default)
            rawValue = performParseTimeClamp(rawValue);

        if (!Validator::isValid(rawValue, options))
            return std::nullopt;

        range.consumeIncludingWhitespace();
        return rawValue;
    }
};

// Shared consumer for `Number` tokens for use by dimensional primitives that support "unitless" values.
template<typename Primitive, typename Validator, auto unit> struct NumberConsumerForUnitlessValues {
    static constexpr CSSParserTokenType tokenType = NumberToken;

    static std::optional<typename Primitive::Raw> consume(CSSParserTokenRange& range, const CSSParserContext&, CSSCalcSymbolsAllowed, CSSPropertyParserOptions options)
    {
        ASSERT(range.peek().type() == NumberToken);

        auto numericValue = range.peek().numericValue();
        if (!shouldAcceptUnitlessValue(numericValue, options))
            return std::nullopt;

        auto rawValue = typename Primitive::Raw { unit, numericValue };

        if constexpr (rawValue.range.options != CSS::RangeOptions::Default)
            rawValue = performParseTimeClamp(rawValue);

        if (!Validator::isValid(rawValue, options))
            return std::nullopt;

        range.consumeIncludingWhitespace();
        return rawValue;
    }
};

// Shared consumer for `Function` tokens that processes `calc()` for the provided primitive.
template<typename Primitive> struct FunctionConsumerForCalcValues {
    static constexpr CSSParserTokenType tokenType = FunctionToken;

    static std::optional<typename Primitive::Calc> consume(CSSParserTokenRange& range, const CSSParserContext& context, CSSCalcSymbolsAllowed symbolsAllowed, CSSPropertyParserOptions options)
    {
        ASSERT(range.peek().type() == FunctionToken);

        auto rangeCopy = range;
        if (RefPtr value = CSSCalcValue::parse(rangeCopy, context, Primitive::category, Primitive::range, WTFMove(symbolsAllowed), options)) {
            range = rangeCopy;
            return {{ value.releaseNonNull() }};
        }

        return std::nullopt;
    }
};

template<typename T> struct KeywordConsumer {
    static constexpr CSSParserTokenType tokenType = IdentToken;

    static std::optional<T> consume(CSSParserTokenRange& range, const CSSParserContext&, CSSCalcSymbolsAllowed, CSSPropertyParserOptions)
    {
        ASSERT(range.peek().type() == IdentToken);

        if (range.peek().id() == T::value) {
            range.consumeIncludingWhitespace();
            return T { };
        }

        return std::nullopt;
    }
};

} // namespace CSSPropertyParserHelpers
} // namespace WebCore
