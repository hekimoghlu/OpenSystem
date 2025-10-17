/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 28, 2021.
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
#include "CSSPropertyParserConsumer+Sizing.h"

#include "CSSParserContext.h"
#include "CSSParserTokenRange.h"
#include "CSSPrimitiveValue.h"
#include "CSSPropertyParserConsumer+Ident.h"
#include "CSSPropertyParserConsumer+Length.h"
#include "CSSPropertyParserConsumer+Number.h"
#include "CSSPropertyParserConsumer+Primitives.h"
#include "CSSValueKeywords.h"
#include "CSSValueList.h"
#include "CSSValuePair.h"

namespace WebCore {
namespace CSSPropertyParserHelpers {

static RefPtr<CSSValueList> consumeAspectRatioValue(CSSParserTokenRange& range, const CSSParserContext& context)
{
    auto leftValue = consumeNumber(range, context, ValueRange::NonNegative);
    if (!leftValue)
        return nullptr;

    auto rightValue = consumeSlashIncludingWhitespace(range)
        ? consumeNumber(range, context, ValueRange::NonNegative)
        : CSSPrimitiveValue::create(1);
    if (!rightValue)
        return nullptr;

    return CSSValueList::createSlashSeparated(leftValue.releaseNonNull(), rightValue.releaseNonNull());
}

RefPtr<CSSValue> consumeAspectRatio(CSSParserTokenRange& range, const CSSParserContext& context)
{
    // <'aspect-ratio'> = auto || <ratio>
    // https://drafts.csswg.org/css-sizing-4/#aspect-ratio

    RefPtr<CSSPrimitiveValue> autoValue;
    if (range.peek().type() == IdentToken)
        autoValue = consumeIdent<CSSValueAuto>(range);
    if (range.atEnd())
        return autoValue;
    auto ratioList = consumeAspectRatioValue(range, context);
    if (!ratioList)
        return nullptr;
    if (!autoValue) {
        autoValue = consumeIdent<CSSValueAuto>(range);
        if (!autoValue)
            return ratioList;
    }
    return CSSValueList::createSpaceSeparated(autoValue.releaseNonNull(), ratioList.releaseNonNull());
}

RefPtr<CSSValue> consumeContainIntrinsicSize(CSSParserTokenRange& range, const CSSParserContext& context)
{
    // <contain-intrinsic-size> = auto? [ none | <length> ]
    // https://drafts.csswg.org/css-sizing-4/#intrinsic-size-override

    RefPtr<CSSPrimitiveValue> autoValue;
    if ((autoValue = consumeIdent<CSSValueAuto>(range))) {
        if (range.atEnd())
            return nullptr;
    }

    if (auto noneValue = consumeIdent<CSSValueNone>(range)) {
        if (autoValue)
            return CSSValuePair::create(autoValue.releaseNonNull(), noneValue.releaseNonNull());
        return noneValue;
    }

    if (auto lengthValue = consumeLength(range, context, HTMLStandardMode, ValueRange::NonNegative)) {
        if (autoValue)
            return CSSValuePair::create(autoValue.releaseNonNull(), lengthValue.releaseNonNull());
        return lengthValue;
    }
    return nullptr;
}

} // namespace CSSPropertyParserHelpers
} // namespace WebCore
