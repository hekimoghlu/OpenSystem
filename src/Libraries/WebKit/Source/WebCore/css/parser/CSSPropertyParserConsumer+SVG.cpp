/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 24, 2022.
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
#include "CSSPropertyParserConsumer+SVG.h"

#include "CSSParserContext.h"
#include "CSSParserTokenRange.h"
#include "CSSPrimitiveValue.h"
#include "CSSPropertyParserConsumer+Color.h"
#include "CSSPropertyParserConsumer+Ident.h"
#include "CSSPropertyParserConsumer+LengthPercentage.h"
#include "CSSPropertyParserConsumer+Number.h"
#include "CSSPropertyParserConsumer+Primitives.h"
#include "CSSPropertyParserConsumer+URL.h"
#include "CSSValueKeywords.h"
#include "CSSValueList.h"

namespace WebCore {
namespace CSSPropertyParserHelpers {

RefPtr<CSSValue> consumePaint(CSSParserTokenRange& range, const CSSParserContext& context)
{
    // <paint> = none | <color> | <url> [none | <color>]? | context-fill | context-stroke
    // https://svgwg.org/svg2-draft/painting.html#SpecifyingStrokePaint

    if (range.peek().id() == CSSValueNone)
        return consumeIdent(range);
    auto url = consumeURL(range);
    if (url) {
        RefPtr<CSSValue> parsedValue;
        if (range.peek().id() == CSSValueNone)
            parsedValue = consumeIdent(range);
        else
            parsedValue = consumeColor(range, context);
        if (parsedValue)
            return CSSValueList::createSpaceSeparated(url.releaseNonNull(), parsedValue.releaseNonNull());
        return url;
    }
    return consumeColor(range, context);
}

RefPtr<CSSValue> consumePaintOrder(CSSParserTokenRange& range, const CSSParserContext&)
{
    // <'paint-order'> = normal | [ fill || stroke || markers ]
    // https://svgwg.org/svg2-draft/painting.html#PaintOrderProperty

    if (range.peek().id() == CSSValueNormal)
        return consumeIdent(range);

    Vector<CSSValueID, 3> paintTypeList;
    RefPtr<CSSPrimitiveValue> fill;
    RefPtr<CSSPrimitiveValue> stroke;
    RefPtr<CSSPrimitiveValue> markers;
    do {
        CSSValueID id = range.peek().id();
        if (id == CSSValueFill && !fill)
            fill = consumeIdent(range);
        else if (id == CSSValueStroke && !stroke)
            stroke = consumeIdent(range);
        else if (id == CSSValueMarkers && !markers)
            markers = consumeIdent(range);
        else
            return nullptr;
        paintTypeList.append(id);
    } while (!range.atEnd());

    // After parsing we serialize the paint-order list. Since it is not possible to
    // pop a last list items from CSSValueList without bigger cost, we create the
    // list after parsing.
    CSSValueID firstPaintOrderType = paintTypeList.at(0);
    CSSValueListBuilder paintOrderList;
    switch (firstPaintOrderType) {
    case CSSValueFill:
    case CSSValueStroke:
        paintOrderList.append(firstPaintOrderType == CSSValueFill ? fill.releaseNonNull() : stroke.releaseNonNull());
        if (paintTypeList.size() > 1) {
            if (paintTypeList.at(1) == CSSValueMarkers)
                paintOrderList.append(markers.releaseNonNull());
        }
        break;
    case CSSValueMarkers:
        paintOrderList.append(markers.releaseNonNull());
        if (paintTypeList.size() > 1) {
            if (paintTypeList.at(1) == CSSValueStroke)
                paintOrderList.append(stroke.releaseNonNull());
        }
        break;
    default:
        ASSERT_NOT_REACHED();
        return nullptr;
    }

    return CSSValueList::createSpaceSeparated(WTFMove(paintOrderList));
}

RefPtr<CSSValue> consumeStrokeDasharray(CSSParserTokenRange& range, const CSSParserContext& context)
{
    // <'stroke-dasharray'> = none | [ [ <length-percentage> | <number> ]+ ]#
    // https://svgwg.org/svg2-draft/painting.html#StrokeDashing

    CSSValueID id = range.peek().id();
    if (id == CSSValueNone)
        return consumeIdent(range);
    CSSValueListBuilder dashes;
    do {
        RefPtr<CSSPrimitiveValue> dash = consumeLengthPercentage(range, context, HTMLStandardMode, ValueRange::NonNegative, UnitlessQuirk::Forbid, UnitlessZeroQuirk::Forbid);
        if (!dash)
            dash = consumeNumber(range, context, ValueRange::NonNegative);
        if (!dash || (consumeCommaIncludingWhitespace(range) && range.atEnd()))
            return nullptr;
        dashes.append(dash.releaseNonNull());
    } while (!range.atEnd());
    return CSSValueList::createCommaSeparated(WTFMove(dashes));
}

} // namespace CSSPropertyParserHelpers
} // namespace WebCore
