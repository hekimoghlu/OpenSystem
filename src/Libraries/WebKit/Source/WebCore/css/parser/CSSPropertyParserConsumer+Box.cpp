/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 15, 2022.
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
#include "CSSPropertyParserConsumer+Box.h"

#include "CSSCalcSymbolTable.h"
#include "CSSParserContext.h"
#include "CSSParserTokenRange.h"
#include "CSSPrimitiveValue.h"
#include "CSSPropertyParserConsumer+Ident.h"
#include "CSSPropertyParserConsumer+LengthPercentage.h"
#include "CSSValueKeywords.h"
#include "CSSValueList.h"

namespace WebCore {
namespace CSSPropertyParserHelpers {

RefPtr<CSSValue> consumeMarginPhysical(CSSParserTokenRange& range, const CSSParserContext& context, CSSPropertyID currentShorthand)
{
    // <margin-physical> = <length-percentage> | auto
    // https://drafts.csswg.org/css-box/#margin-physical

    if (range.peek().id() == CSSValueAuto)
        return consumeIdent(range);

    auto unitless = currentShorthand != CSSPropertyInset ? UnitlessQuirk::Allow : UnitlessQuirk::Forbid;
    auto anchorSizePolicy = context.propertySettings.cssAnchorPositioningEnabled ? AnchorSizePolicy::Allow : AnchorSizePolicy::Forbid;
    return consumeLengthPercentage(range, context, ValueRange::All, unitless, UnitlessZeroQuirk::Allow, AnchorPolicy::Forbid, anchorSizePolicy);
}

RefPtr<CSSValue> consumeMarginTrim(CSSParserTokenRange& range, const CSSParserContext&)
{
    // <'margin-trim'> = none | [ block || inline ] | [ block-start || inline-start || block-end || inline-end ]
    // https://drafts.csswg.org/css-box/#margin-trim

    auto firstValue = range.peek().id();
    if (firstValue == CSSValueBlock || firstValue == CSSValueInline || firstValue == CSSValueNone)
        return consumeIdent(range).releaseNonNull();
    Vector<CSSValueID, 4> idents;
    while (auto ident = consumeIdentRaw<CSSValueBlockStart, CSSValueBlockEnd, CSSValueInlineStart, CSSValueInlineEnd>(range)) {
        if (idents.contains(*ident))
            return nullptr;
        idents.append(*ident);
    }
    // Try to serialize into either block or inline form
    if (idents.size() == 2) {
        if (idents.contains(CSSValueBlockStart) && idents.contains(CSSValueBlockEnd))
            return CSSPrimitiveValue::create(CSSValueBlock);
        if (idents.contains(CSSValueInlineStart) && idents.contains(CSSValueInlineEnd))
            return CSSPrimitiveValue::create(CSSValueInline);
    }
    CSSValueListBuilder list;
    for (auto ident : idents)
        list.append(CSSPrimitiveValue::create(ident));
    return CSSValueList::createSpaceSeparated(WTFMove(list));
}

} // namespace CSSPropertyParserHelpers
} // namespace WebCore
