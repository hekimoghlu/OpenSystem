/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 3, 2023.
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
#include "CSSPropertyParserConsumer+Inline.h"

#include "CSSLineBoxContainValue.h"
#include "CSSParserContext.h"
#include "CSSParserTokenRange.h"
#include "CSSPrimitiveValue.h"
#include "CSSPropertyParserConsumer+Ident.h"
#include "CSSPropertyParserConsumer+Number.h"
#include "CSSValueKeywords.h"
#include "CSSValuePair.h"

namespace WebCore {
namespace CSSPropertyParserHelpers {

static RefPtr<CSSValue> consumeTextEdge(CSSParserTokenRange& range)
{
    // <text-edge> = [ text | cap | ex | ideographic | ideographic-ink ]
    //               [ text | alphabetic | ideographic | ideographic-ink ]?
    // https://drafts.csswg.org/css-inline-3/#typedef-text-edge

    auto firstValue = consumeIdent<CSSValueText, CSSValueCap, CSSValueEx, CSSValueIdeographic, CSSValueIdeographicInk>(range);
    if (!firstValue)
        return nullptr;

    auto secondValue = consumeIdent<CSSValueText, CSSValueAlphabetic, CSSValueIdeographic, CSSValueIdeographicInk>(range);

    // https://drafts.csswg.org/css-inline-3/#text-edges
    // "If only one value is specified, both edges are assigned that same keyword if possible; else text is assumed as the missing value."
    auto shouldSerializeSecondValue = [&]() {
        if (!secondValue)
            return false;
        if (firstValue->valueID() == CSSValueCap || firstValue->valueID() == CSSValueEx)
            return secondValue->valueID() != CSSValueText;
        return firstValue->valueID() != secondValue->valueID();
    }();
    if (!shouldSerializeSecondValue)
        return firstValue;

    return CSSValuePair::create(firstValue.releaseNonNull(), secondValue.releaseNonNull());
}

RefPtr<CSSValue> consumeLineFitEdge(CSSParserTokenRange& range, const CSSParserContext&)
{
    // <'line-fit-edge'> = leading | <text-edge>
    // https://drafts.csswg.org/css-inline-3/#propdef-line-fit-edge

    if (range.peek().id() == CSSValueLeading)
        return consumeIdent(range);
    return consumeTextEdge(range);
}

RefPtr<CSSValue> consumeTextBoxEdge(CSSParserTokenRange& range, const CSSParserContext&)
{
    // <'text-box-edge'> = auto | <text-edge>
    // https://drafts.csswg.org/css-inline-3/#propdef-text-box-edge

    if (range.peek().id() == CSSValueAuto)
        return consumeIdent(range);
    return consumeTextEdge(range);
}

RefPtr<CSSValue> consumeWebkitInitialLetter(CSSParserTokenRange& range, const CSSParserContext& context)
{
    // Standard says this should be:
    //
    // <'initial-letter'> = normal | <number [1,âˆž]> <integer [1,âˆž]> | <number [1,âˆž]> && [ drop | raise ]?
    // https://drafts.csswg.org/css-inline-3/#sizing-drop-initials

    if (auto ident = consumeIdent<CSSValueNormal>(range))
        return ident;
    auto height = consumeNumber(range, context, ValueRange::NonNegative);
    if (!height)
        return nullptr;
    RefPtr<CSSPrimitiveValue> position;
    if (!range.atEnd()) {
        position = consumeNumber(range, context, ValueRange::NonNegative);
        if (!position || !range.atEnd())
            return nullptr;
    } else
        position = height.copyRef();
    return CSSValuePair::create(position.releaseNonNull(), height.releaseNonNull());
}

RefPtr<CSSValue> consumeWebkitLineBoxContain(CSSParserTokenRange& range, const CSSParserContext&)
{
    // <'-webkit-line-box-contain'> = none | [ block || inline || font || glyphs || replaced || inline-box || initial-letter ]
    // NOTE: This is a non-standard property with no standard equivalent.

    if (range.peek().id() == CSSValueNone)
        return consumeIdent(range);

    OptionSet<LineBoxContain> value;
    while (range.peek().type() == IdentToken) {
        auto flag = [&]() -> std::optional<LineBoxContain> {
            switch (range.peek().id()) {
            case CSSValueBlock:
                return LineBoxContain::Block;
            case CSSValueInline:
                return LineBoxContain::Inline;
            case CSSValueFont:
                return LineBoxContain::Font;
            case CSSValueGlyphs:
                return LineBoxContain::Glyphs;
            case CSSValueReplaced:
                return LineBoxContain::Replaced;
            case CSSValueInlineBox:
                return LineBoxContain::InlineBox;
            case CSSValueInitialLetter:
                return LineBoxContain::InitialLetter;
            default:
                return std::nullopt;
            }
        }();
        if (!flag || value.contains(*flag))
            return nullptr;
        value.add(flag);
        range.consumeIncludingWhitespace();
    }
    if (!value)
        return nullptr;
    return CSSLineBoxContainValue::create(value);
}

} // namespace CSSPropertyParserHelpers
} // namespace WebCore
