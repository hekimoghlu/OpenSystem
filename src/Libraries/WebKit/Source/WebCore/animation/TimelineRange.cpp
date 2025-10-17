/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 6, 2021.
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
#include "TimelineRange.h"
#include "CSSNumericFactory.h"
#include "CSSPropertyParserConsumer+Timeline.h"
#include "CSSValuePair.h"
#include "CSSValuePool.h"
#include "StyleBuilderConverter.h"
#include "StyleBuilderState.h"

namespace WebCore {

static bool percentIsDefault(float percent, SingleTimelineRange::Type type)
{
    return percent == SingleTimelineRange::defaultValue(type).value();
}

bool SingleTimelineRange::isDefault(const Length& offset, Type type)
{
    return offset.isAuto() || offset.isNormal() || (offset.isPercent() && percentIsDefault(offset.percent(), type));
}

bool SingleTimelineRange::isDefault(const CSSPrimitiveValue& value, Type type)
{
    return value.valueID() == CSSValueNormal || (value.isPercentage() && !value.isCalculated() && percentIsDefault(value.resolveAsPercentageNoConversionDataRequired(), type));
}

Length SingleTimelineRange::defaultValue(Type type)
{
    return type == Type::Start ? Length(0, LengthType::Percent) : Length(100, LengthType::Percent);
}

bool SingleTimelineRange::isOffsetValue(const CSSPrimitiveValue& value)
{
    return value.isLength() || value.isPercentage() || value.isCalculatedPercentageWithLength();
}

TimelineRangeValue SingleTimelineRange::serialize() const
{
    if (name == Name::Normal)
        return CSSPrimitiveValue::create(valueID(name))->stringValue();
    return TimelineRangeOffset { CSSPrimitiveValue::create(valueID(name))->stringValue(), offset.isPercentOrCalculated() ? CSSNumericFactory::percent(offset.percent()) : CSSNumericFactory::px(offset.value()) };
}

static const std::optional<CSSToLengthConversionData> cssToLengthConversionData(RefPtr<Element> element)
{
    CheckedPtr elementRenderer = element->renderer();
    if (!elementRenderer)
        return std::nullopt;
    CheckedPtr elementParentRenderer = elementRenderer->parent();
    Ref document = element->document();
    CheckedPtr documentElement = document->documentElement();
    if (!documentElement)
        return std::nullopt;

    // FIXME: Investigate container query units
    return CSSToLengthConversionData {
        elementRenderer->style(),
        documentElement->renderer() ? &documentElement->renderer()->style() : nullptr,
        elementParentRenderer ? &elementParentRenderer->style() : nullptr,
        document->renderView()
    };
}

Length SingleTimelineRange::lengthForCSSValue(RefPtr<const CSSPrimitiveValue> value, RefPtr<Element> element)
{
    if (!value || !element)
        return { };
    if (value->valueID() == CSSValueAuto)
        return { };

    auto conversionData = cssToLengthConversionData(element);
    if (!conversionData) {
        if (value->isPercentage())
            return Length(value->resolveAsPercentageNoConversionDataRequired(), LengthType::Percent);
        if (value->isPx())
            return Length(value->resolveAsLengthNoConversionDataRequired(), LengthType::Fixed);
        return { };
    }

    if (value->isLength())
        return value->resolveAsLength<WebCore::Length>(*conversionData);

    if (value->isPercentage())
        return Length(value->resolveAsPercentage(*conversionData), LengthType::Percent);

    if (value->isCalculatedPercentageWithLength()) {
        if (RefPtr cssCalcValue = value->cssCalcValue())
            return Length(cssCalcValue->createCalculationValue(*conversionData, CSSCalcSymbolTable { }));
    }

    ASSERT_NOT_REACHED();
    return { };
}

SingleTimelineRange SingleTimelineRange::range(const CSSValue& value, Type type, Style::BuilderState* state, RefPtr<Element> element)
{
    if (RefPtr primitiveValue = dynamicDowncast<CSSPrimitiveValue>(value)) {
        // <length-percentage>
        if (SingleTimelineRange::isOffsetValue(*primitiveValue))
            return { SingleTimelineRange::Name::Omitted, state ? Style::BuilderConverter::convertLength(*state, *primitiveValue) : lengthForCSSValue(primitiveValue, element) };
        // <timeline-range-name> or Normal
        return { SingleTimelineRange::timelineName(primitiveValue->valueID()), defaultValue(type) };
    }
    RefPtr pair = dynamicDowncast<CSSValuePair>(value);
    if (!pair)
        return { };

    // <timeline-range-name> <length-percentage>
    Ref primitiveValue = downcast<CSSPrimitiveValue>(pair->second());
    ASSERT(SingleTimelineRange::isOffsetValue(primitiveValue));

    return { SingleTimelineRange::timelineName(pair->first().valueID()), state ? Style::BuilderConverter::convertLength(*state, primitiveValue) : lengthForCSSValue(RefPtr { primitiveValue.ptr() }, element) };
}

RefPtr<CSSValue> SingleTimelineRange::parse(TimelineRangeValue&& value, RefPtr<Element> element, Type type)
{
    if (!element)
        return { };
    RefPtr document = element->protectedDocument();
    if (!document)
        return { };
    const auto& parserContext = document->cssParserContext();
    return WTF::switchOn(value,
    [&](String& rangeString) -> RefPtr<CSSValue> {
        CSSTokenizer tokenizer(rangeString);
        auto tokenRange = tokenizer.tokenRange();
        tokenRange.consumeWhitespace();
        return CSSPropertyParserHelpers::consumeAnimationRange(tokenRange, parserContext, type);
    },
    [&](TimelineRangeOffset& rangeOffset) -> RefPtr<CSSValue> {
        CSSTokenizer tokenizer(rangeOffset.rangeName);
        auto tokenRange = tokenizer.tokenRange();
        tokenRange.consumeWhitespace();
        if (auto consumedRangeName = CSSPropertyParserHelpers::consumeAnimationRange(tokenRange, parserContext, type)) {
            if (rangeOffset.offset)
                return CSSValuePair::createNoncoalescing(*consumedRangeName, *rangeOffset.offset->toCSSValue());
            return consumedRangeName;
        }
        if (RefPtr offset = rangeOffset.offset)
            return offset->toCSSValue();
        return nullptr;
    },
    [&](RefPtr<CSSKeywordValue> rangeKeyword) {
        return rangeKeyword->toCSSValue();
    },
    [&](RefPtr<CSSNumericValue> rangeValue) {
        return rangeValue->toCSSValue();
    });
}

SingleTimelineRange::Name SingleTimelineRange::timelineName(CSSValueID valueID)
{
    switch (valueID) {
    case CSSValueNormal:
        return SingleTimelineRange::Name::Normal;
    case CSSValueCover:
        return SingleTimelineRange::Name::Cover;
    case CSSValueContain:
        return SingleTimelineRange::Name::Contain;
    case CSSValueEntry:
        return SingleTimelineRange::Name::Entry;
    case CSSValueExit:
        return SingleTimelineRange::Name::Exit;
    case CSSValueEntryCrossing:
        return SingleTimelineRange::Name::EntryCrossing;
    case CSSValueExitCrossing:
        return SingleTimelineRange::Name::ExitCrossing;
    default:
        ASSERT_NOT_REACHED();
        return SingleTimelineRange::Name::Normal;
    }
}

CSSValueID SingleTimelineRange::valueID(SingleTimelineRange::Name range)
{
    switch (range) {
    case SingleTimelineRange::Name::Normal:
        return CSSValueNormal;
    case SingleTimelineRange::Name::Cover:
        return CSSValueCover;
    case SingleTimelineRange::Name::Contain:
        return CSSValueContain;
    case SingleTimelineRange::Name::Entry:
        return CSSValueEntry;
    case SingleTimelineRange::Name::Exit:
        return CSSValueExit;
    case SingleTimelineRange::Name::EntryCrossing:
        return CSSValueEntryCrossing;
    case SingleTimelineRange::Name::ExitCrossing:
        return CSSValueExitCrossing;
    case SingleTimelineRange::Name::Omitted:
        return CSSValueInvalid;
    }
    ASSERT_NOT_REACHED();
    return CSSValueNormal;
}

WTF::TextStream& operator<<(WTF::TextStream& ts, const SingleTimelineRange& range)
{
    switch (range.name) {
    case SingleTimelineRange::Name::Normal: ts << "Normal "; break;
    case SingleTimelineRange::Name::Omitted: ts << "Omitted "; break;
    case SingleTimelineRange::Name::Cover: ts << "Cover "; break;
    case SingleTimelineRange::Name::Contain: ts << "Contain "; break;
    case SingleTimelineRange::Name::Entry: ts << "Entry "; break;
    case SingleTimelineRange::Name::Exit: ts << "Exit "; break;
    case SingleTimelineRange::Name::EntryCrossing: ts << "EntryCrossing "; break;
    case SingleTimelineRange::Name::ExitCrossing: ts << "ExitCrossing "; break;
    }
    ts << range.offset;
    return ts;
}

TimelineRange TimelineRange::defaultForScrollTimeline()
{
    return {
        { SingleTimelineRange::Name::Omitted, SingleTimelineRange::defaultValue(SingleTimelineRange::Type::Start) },
        { SingleTimelineRange::Name::Omitted, SingleTimelineRange::defaultValue(SingleTimelineRange::Type::End) }
    };
}

TimelineRange TimelineRange::defaultForViewTimeline()
{
    return {
        { SingleTimelineRange::Name::Cover, SingleTimelineRange::defaultValue(SingleTimelineRange::Type::Start) },
        { SingleTimelineRange::Name::Cover, SingleTimelineRange::defaultValue(SingleTimelineRange::Type::End) }
    };
}

} // namespace WebCore
