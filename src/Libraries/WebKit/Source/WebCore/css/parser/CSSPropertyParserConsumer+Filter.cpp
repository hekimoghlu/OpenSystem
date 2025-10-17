/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 3, 2023.
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
#include "CSSPropertyParserConsumer+Filter.h"

#include "CSSAppleColorFilterProperty.h"
#include "CSSAppleColorFilterPropertyValue.h"
#include "CSSFilterFunctionDescriptor.h"
#include "CSSFilterProperty.h"
#include "CSSFilterPropertyValue.h"
#include "CSSParserContext.h"
#include "CSSParserTokenRange.h"
#include "CSSPrimitiveValue.h"
#include "CSSPropertyParserConsumer+AngleDefinitions.h"
#include "CSSPropertyParserConsumer+Background.h"
#include "CSSPropertyParserConsumer+Color.h"
#include "CSSPropertyParserConsumer+Ident.h"
#include "CSSPropertyParserConsumer+LengthDefinitions.h"
#include "CSSPropertyParserConsumer+MetaConsumer.h"
#include "CSSPropertyParserConsumer+NumberDefinitions.h"
#include "CSSPropertyParserConsumer+PercentageDefinitions.h"
#include "CSSPropertyParserConsumer+Primitives.h"
#include "CSSPropertyParserConsumer+URL.h"
#include "CSSToLengthConversionData.h"
#include "CSSTokenizer.h"
#include "CSSValueKeywords.h"
#include "FilterOperations.h"
#include "StyleFilterProperty.h"
#include <wtf/text/StringView.h>

namespace WebCore {
namespace CSSPropertyParserHelpers {

template<CSSValueID filterFunction> static decltype(auto) consumeNumberOrPercentFilterParameter(CSSParserTokenRange& args, const CSSParserContext& context)
{
    if constexpr (filterFunctionAllowsValuesGreaterThanOne<filterFunction>()) {
        return MetaConsumer<
            CSS::Number<CSS::Nonnegative>,
            CSS::Percentage<CSS::Nonnegative>
        >::consume(args, context, { }, { .parserMode = context.mode });
    } else {
        return MetaConsumer<
            CSS::Number<CSS::ClosedUnitRangeClampUpper>,
            CSS::Percentage<CSS::ClosedPercentageRangeClampUpper>
        >::consume(args, context, { }, { .parserMode = context.mode });
    }
}

static std::optional<CSS::AppleInvertLightnessFunction> consumeFilterAppleInvertLightness(CSSParserTokenRange& range, const CSSParserContext&)
{
    // <-apple-invert-lightness()> = -apple-invert-lightness()
    // Non-standard

    auto args = consumeFunction(range);
    if (!args.atEnd())
        return { };

    return CSS::AppleInvertLightnessFunction { .parameters = { } };
}

static std::optional<CSS::BlurFunction> consumeFilterBlur(CSSParserTokenRange& range, const CSSParserContext& context)
{
    // blur() = blur( <length [0,âˆž]>? )
    // https://drafts.fxtf.org/filter-effects/#funcdef-filter-blur

    auto args = consumeFunction(range);
    if (args.atEnd())
        return { CSS::BlurFunction { .parameters = { } } };

    const auto lengthOptions = CSSPropertyParserOptions {
        .parserMode = context.mode,
        .unitlessZero = UnitlessZeroQuirk::Allow
    };
    auto parsedValue = MetaConsumer<CSS::Length<CSS::Nonnegative>>::consume(args, context, { }, lengthOptions);
    if (!parsedValue || !args.atEnd())
        return { };

    return CSS::BlurFunction { .parameters = { CSS::Blur::Parameter { WTFMove(*parsedValue) } } };
}

static std::optional<CSS::BrightnessFunction> consumeFilterBrightness(CSSParserTokenRange& range, const CSSParserContext& context)
{
    // brightness() = brightness( [ <number [0,âˆž]> | <percentage [0,âˆž]> ]? )
    // https://drafts.fxtf.org/filter-effects/#funcdef-filter-brightness

    auto args = consumeFunction(range);
    if (args.atEnd())
        return CSS::BrightnessFunction { .parameters = { } };

    auto parsedValue = consumeNumberOrPercentFilterParameter<CSS::BrightnessFunction::name>(args, context);
    if (!parsedValue || !args.atEnd())
        return { };

    return CSS::BrightnessFunction { .parameters = { CSS::Brightness::Parameter { WTFMove(*parsedValue) } } };
}

static std::optional<CSS::ContrastFunction> consumeFilterContrast(CSSParserTokenRange& range, const CSSParserContext& context)
{
    // contrast() = contrast( [ <number [0,âˆž]> | <percentage [0,âˆž]> ]? )
    // https://drafts.fxtf.org/filter-effects/#funcdef-filter-contrast

    auto args = consumeFunction(range);
    if (args.atEnd())
        return CSS::ContrastFunction { .parameters = { } };

    auto parsedValue = consumeNumberOrPercentFilterParameter<CSS::ContrastFunction::name>(args, context);
    if (!parsedValue || !args.atEnd())
        return { };

    return CSS::ContrastFunction { .parameters = { CSS::Contrast::Parameter { WTFMove(*parsedValue) } } };
}

static std::optional<CSS::DropShadowFunction> consumeFilterDropShadow(CSSParserTokenRange& range, const CSSParserContext& context)
{
    // drop-shadow() = drop-shadow( [ <color>? && [<length>{2} <length [0,âˆž]>?] ] )
    // https://drafts.fxtf.org/filter-effects/#funcdef-filter-drop-shadow

    auto args = consumeFunction(range);

    const auto lengthOptions = CSSPropertyParserOptions {
        .parserMode = context.mode,
        .unitlessZero = UnitlessZeroQuirk::Allow
    };

    std::optional<CSS::Color> color;
    std::optional<CSS::Length<>> x;
    std::optional<CSS::Length<>> y;
    std::optional<CSS::Length<CSS::Nonnegative>> stdDeviation;

    auto consumeOptionalColor = [&] -> bool {
        if (color)
            return false;
        auto maybeColor = consumeUnresolvedColor(args, context);
        if (!maybeColor)
            return false;
        color = WTFMove(*maybeColor);
        return true;
    };

    auto consumeLengths = [&] -> bool {
        if (x)
            return false;
        x = MetaConsumer<CSS::Length<>>::consume(args, context, { }, lengthOptions);
        if (!x)
            return false;
        y = MetaConsumer<CSS::Length<>>::consume(args, context, { }, lengthOptions);
        if (!y)
            return false;

        stdDeviation = MetaConsumer<CSS::Length<CSS::Nonnegative>>::consume(args, context, { }, lengthOptions);
        return true;
    };

    while (!args.atEnd()) {
        if (consumeOptionalColor() || consumeLengths())
            continue;
        break;
    }

    if (!y || !args.atEnd())
        return { };

    return CSS::DropShadowFunction {
        .parameters = {
            .color = WTFMove(color),
            .location = { WTFMove(*x), WTFMove(*y) },
            .stdDeviation = WTFMove(stdDeviation)
        }
    };
}

static std::optional<CSS::GrayscaleFunction> consumeFilterGrayscale(CSSParserTokenRange& range, const CSSParserContext& context)
{
    // grayscale() = grayscale( [ <number [0,1(clamp upper)] > | <percentage [0,100(clamp upper)]> ]? )
    // https://drafts.fxtf.org/filter-effects/#funcdef-filter-grayscale

    auto args = consumeFunction(range);
    if (args.atEnd())
        return CSS::GrayscaleFunction { .parameters = { } };

    auto parsedValue = consumeNumberOrPercentFilterParameter<CSS::GrayscaleFunction::name>(args, context);
    if (!parsedValue || !args.atEnd())
        return { };

    return CSS::GrayscaleFunction { .parameters = { CSS::Grayscale::Parameter { WTFMove(*parsedValue) } } };
}

static std::optional<CSS::HueRotateFunction> consumeFilterHueRotate(CSSParserTokenRange& range, const CSSParserContext& context)
{
    // hue-rotate() = hue-rotate( [ <angle> | <zero> ]? )
    // https://drafts.fxtf.org/filter-effects/#funcdef-filter-hue-rotate

    auto args = consumeFunction(range);
    if (args.atEnd())
        return CSS::HueRotateFunction { .parameters = { } };

    const auto angleOptions = CSSPropertyParserOptions {
        .parserMode = context.mode,
        .unitlessZero = UnitlessZeroQuirk::Allow
    };

    auto parsedValue = MetaConsumer<CSS::Angle<>>::consume(args, context, { }, angleOptions);
    if (!parsedValue || !args.atEnd())
        return { };

    return CSS::HueRotateFunction { .parameters = { CSS::HueRotate::Parameter { WTFMove(*parsedValue) } } };
}

static std::optional<CSS::InvertFunction> consumeFilterInvert(CSSParserTokenRange& range, const CSSParserContext& context)
{
    // invert() = invert( [ <number [0,1(clamp upper)] > | <percentage [0,100(clamp upper)]> ]? )
    // https://drafts.fxtf.org/filter-effects/#funcdef-filter-invert

    auto args = consumeFunction(range);
    if (args.atEnd())
        return CSS::InvertFunction { .parameters = { } };

    auto parsedValue = consumeNumberOrPercentFilterParameter<CSS::InvertFunction::name>(args, context);
    if (!parsedValue || !args.atEnd())
        return { };

    return CSS::InvertFunction { .parameters = { CSS::Invert::Parameter { WTFMove(*parsedValue) } } };
}

static std::optional<CSS::OpacityFunction> consumeFilterOpacity(CSSParserTokenRange& range, const CSSParserContext& context)
{
    // opacity() = opacity( [ <number [0,1(clamp upper)] > | <percentage [0,100(clamp upper)]> ]? )
    // https://drafts.fxtf.org/filter-effects/#funcdef-filter-opacity

    auto args = consumeFunction(range);
    if (args.atEnd())
        return CSS::OpacityFunction { .parameters = { } };

    auto parsedValue = consumeNumberOrPercentFilterParameter<CSS::OpacityFunction::name>(args, context);
    if (!parsedValue || !args.atEnd())
        return { };

    return CSS::OpacityFunction { .parameters = { CSS::Opacity::Parameter { WTFMove(*parsedValue) } } };
}

static std::optional<CSS::SaturateFunction> consumeFilterSaturate(CSSParserTokenRange& range, const CSSParserContext& context)
{
    // saturate() = saturate( [ <number [0,âˆž]> | <percentage [0,âˆž]> ]? )
    // https://drafts.fxtf.org/filter-effects/#funcdef-filter-saturate

    auto args = consumeFunction(range);
    if (args.atEnd())
        return CSS::SaturateFunction { .parameters = { } };

    auto parsedValue = consumeNumberOrPercentFilterParameter<CSS::SaturateFunction::name>(args, context);
    if (!parsedValue || !args.atEnd())
        return { };

    return CSS::SaturateFunction { .parameters = { CSS::Saturate::Parameter { WTFMove(*parsedValue) } } };
}

static std::optional<CSS::SepiaFunction> consumeFilterSepia(CSSParserTokenRange& range, const CSSParserContext& context)
{
    // sepia() = sepia( [ <number [0,1(clamp upper)] > | <percentage [0,100(clamp upper)]> ]? )
    // https://drafts.fxtf.org/filter-effects/#funcdef-filter-sepia

    auto args = consumeFunction(range);
    if (args.atEnd())
        return CSS::SepiaFunction { .parameters = { } };

    auto parsedValue = consumeNumberOrPercentFilterParameter<CSS::SepiaFunction::name>(args, context);
    if (!parsedValue || !args.atEnd())
        return { };

    return CSS::SepiaFunction { .parameters = { CSS::Sepia::Parameter { WTFMove(*parsedValue) } } };
}

static std::optional<CSS::FilterProperty::List> consumeUnresolvedFilterValueList(CSSParserTokenRange& range, const CSSParserContext& context)
{
    // <filter-value-list> = [ <filter-function> | <url> ]+
    // <filter-function> = <blur()> | <brightness()> | <contrast()> | <drop-shadow()> | <grayscale()> | <hue-rotate()> | <invert()> | <opacity()> | <sepia()> | <saturate()>
    // https://drafts.fxtf.org/filter-effects/#typedef-filter-value-list

    auto rangeCopy = range;

    CSS::FilterProperty::List list;

    auto appendOnSuccess = [&](auto&& value) -> bool {
        if (!value)
            return false;
        list.value.append(WTFMove(*value));
        return true;
    };

    do {
        if (auto url = consumeURLRaw(rangeCopy)) {
            list.value.append(CSS::FilterReference { url.toString() });
            continue;
        }

        switch (rangeCopy.peek().functionId()) {
        case CSSValueBlur:
            if (!appendOnSuccess(consumeFilterBlur(rangeCopy, context)))
                return { };
            break;
        case CSSValueBrightness:
            if (!appendOnSuccess(consumeFilterBrightness(rangeCopy, context)))
                return { };
            break;
        case CSSValueContrast:
            if (!appendOnSuccess(consumeFilterContrast(rangeCopy, context)))
                return { };
            break;
        case CSSValueDropShadow:
            if (!appendOnSuccess(consumeFilterDropShadow(rangeCopy, context)))
                return { };
            break;
        case CSSValueGrayscale:
            if (!appendOnSuccess(consumeFilterGrayscale(rangeCopy, context)))
                return { };
            break;
        case CSSValueHueRotate:
            if (!appendOnSuccess(consumeFilterHueRotate(rangeCopy, context)))
                return { };
            break;
        case CSSValueInvert:
            if (!appendOnSuccess(consumeFilterInvert(rangeCopy, context)))
                return { };
            break;
        case CSSValueOpacity:
            if (!appendOnSuccess(consumeFilterOpacity(rangeCopy, context)))
                return { };
            break;
        case CSSValueSaturate:
            if (!appendOnSuccess(consumeFilterSaturate(rangeCopy, context)))
                return { };
            break;
        case CSSValueSepia:
            if (!appendOnSuccess(consumeFilterSepia(rangeCopy, context)))
                return { };
            break;
        default:
            return { };
        }
    } while (!rangeCopy.atEnd());

    range = rangeCopy;

    return { WTFMove(list) };
}

std::optional<CSS::FilterProperty> consumeUnresolvedFilter(CSSParserTokenRange& range, const CSSParserContext& context)
{
    if (range.peek().id() == CSSValueNone) {
        range.consumeIncludingWhitespace();
        return CSS::FilterProperty { CSS::Keyword::None { } };
    }
    if (auto filterValueList = consumeUnresolvedFilterValueList(range, context))
        return CSS::FilterProperty { WTFMove(*filterValueList) };
    return { };
}

RefPtr<CSSValue> consumeFilter(CSSParserTokenRange& range, const CSSParserContext& context)
{
    if (auto property = consumeUnresolvedFilter(range, context))
        return CSSFilterPropertyValue::create({ WTFMove(*property) });
    return nullptr;
}

// MARK: - <-apple-color-filter>

static std::optional<CSS::AppleColorFilterProperty::List> consumeUnresolvedAppleColorFilterValueList(CSSParserTokenRange& range, const CSSParserContext& context)
{
    // <-apple-color-filter-value-list = <-apple-color-filter-function>+
    // <-apple-color-filter-function> = <-apple-invert-lightness() | <brightness()> | <contrast()> | <grayscale()> | <hue-rotate()> | <invert()> | <opacity()> | <sepia()> | <saturate()>

    auto rangeCopy = range;

    CSS::AppleColorFilterProperty::List list;

    auto appendOnSuccess = [&](auto&& value) -> bool {
        if (!value)
            return false;
        list.value.append(WTFMove(*value));
        return true;
    };

    do {
        switch (rangeCopy.peek().functionId()) {
        case CSSValueAppleInvertLightness:
            if (!appendOnSuccess(consumeFilterAppleInvertLightness(rangeCopy, context)))
                return { };
            break;
        case CSSValueBrightness:
            if (!appendOnSuccess(consumeFilterBrightness(rangeCopy, context)))
                return { };
            break;
        case CSSValueContrast:
            if (!appendOnSuccess(consumeFilterContrast(rangeCopy, context)))
                return { };
            break;
        case CSSValueGrayscale:
            if (!appendOnSuccess(consumeFilterGrayscale(rangeCopy, context)))
                return { };
            break;
        case CSSValueHueRotate:
            if (!appendOnSuccess(consumeFilterHueRotate(rangeCopy, context)))
                return { };
            break;
        case CSSValueInvert:
            if (!appendOnSuccess(consumeFilterInvert(rangeCopy, context)))
                return { };
            break;
        case CSSValueOpacity:
            if (!appendOnSuccess(consumeFilterOpacity(rangeCopy, context)))
                return { };
            break;
        case CSSValueSaturate:
            if (!appendOnSuccess(consumeFilterSaturate(rangeCopy, context)))
                return { };
            break;
        case CSSValueSepia:
            if (!appendOnSuccess(consumeFilterSepia(rangeCopy, context)))
                return { };
            break;
        default:
            return { };
        }
    } while (!rangeCopy.atEnd());

    range = rangeCopy;

    return { WTFMove(list) };
}

std::optional<CSS::AppleColorFilterProperty> consumeUnresolvedAppleColorFilter(CSSParserTokenRange& range, const CSSParserContext& context)
{
    if (range.peek().id() == CSSValueNone) {
        range.consumeIncludingWhitespace();
        return CSS::AppleColorFilterProperty { CSS::Keyword::None { } };
    }
    if (auto filterValueList = consumeUnresolvedAppleColorFilterValueList(range, context))
        return CSS::AppleColorFilterProperty { WTFMove(*filterValueList) };
    return { };
}

RefPtr<CSSValue> consumeAppleColorFilter(CSSParserTokenRange& range, const CSSParserContext& context)
{
    if (auto property = consumeUnresolvedAppleColorFilter(range, context))
        return CSSAppleColorFilterPropertyValue::create({ WTFMove(*property) });
    return nullptr;
}

std::optional<FilterOperations> parseFilterValueListOrNoneRaw(const String& string, const CSSParserContext& context, const Document& document, RenderStyle& style)
{
    CSSTokenizer tokenizer(string);
    CSSParserTokenRange range(tokenizer.tokenRange());

    // Handle leading whitespace.
    range.consumeWhitespace();

    auto filter = consumeUnresolvedFilter(range, context);
    if (!filter)
        return { };

    // Handle trailing whitespace.
    range.consumeWhitespace();

    if (!range.atEnd())
        return { };

    CSSToLengthConversionData conversionData { style, nullptr, nullptr, nullptr };
    return Style::createFilterOperations(*filter, document, style, conversionData);
}

} // namespace CSSPropertyParserHelpers
} // namespace WebCore
