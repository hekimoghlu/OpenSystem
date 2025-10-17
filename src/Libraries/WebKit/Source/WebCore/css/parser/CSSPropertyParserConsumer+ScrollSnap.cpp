/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 31, 2022.
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
#include "CSSPropertyParserConsumer+ScrollSnap.h"

#include "CSSCalcSymbolTable.h"
#include "CSSParserContext.h"
#include "CSSParserTokenRange.h"
#include "CSSPropertyParserConsumer+Ident.h"
#include "CSSValueList.h"

namespace WebCore {
namespace CSSPropertyParserHelpers {

RefPtr<CSSValue> consumeScrollSnapAlign(CSSParserTokenRange& range, const CSSParserContext&)
{
    // <'scroll-snap-align'> = [ none | start | end | center ]{1,2}
    // https://drafts.csswg.org/css-scroll-snap-1/#scroll-snap-align

    auto firstValue = consumeIdent<CSSValueNone, CSSValueStart, CSSValueCenter, CSSValueEnd>(range);
    if (!firstValue)
        return nullptr;

    auto secondValue = consumeIdent<CSSValueNone, CSSValueStart, CSSValueCenter, CSSValueEnd>(range);
    bool shouldAddSecondValue = secondValue && !secondValue->equals(*firstValue);

    // Only add the second value if it differs from the first so that we produce the canonical
    // serialization of this CSSValueList.
    if (shouldAddSecondValue)
        return CSSValueList::createSpaceSeparated(firstValue.releaseNonNull(), secondValue.releaseNonNull());

    return CSSValueList::createSpaceSeparated(firstValue.releaseNonNull());
}

RefPtr<CSSValue> consumeScrollSnapType(CSSParserTokenRange& range, const CSSParserContext&)
{
    // <'scroll-snap-type'> = none | [ x | y | block | inline | both ] [ mandatory | proximity ]?
    // https://drafts.csswg.org/css-scroll-snap-1/#scroll-snap-type

    auto firstValue = consumeIdent<CSSValueNone, CSSValueX, CSSValueY, CSSValueBlock, CSSValueInline, CSSValueBoth>(range);
    if (!firstValue)
        return nullptr;

    // We only add the second value if it is not the initial value as described in specification
    // so that serialization of this CSSValueList produces the canonical serialization.
    auto secondValue = consumeIdent<CSSValueProximity, CSSValueMandatory>(range);
    if (secondValue && secondValue->valueID() != CSSValueProximity)
        return CSSValueList::createSpaceSeparated(firstValue.releaseNonNull(), secondValue.releaseNonNull());

    return CSSValueList::createSpaceSeparated(firstValue.releaseNonNull());
}

} // namespace CSSPropertyParserHelpers
} // namespace WebCore
