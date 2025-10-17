/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 11, 2023.
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
#include "CSSPropertyParserConsumer+Page.h"

#include "CSSCalcSymbolTable.h"
#include "CSSParserContext.h"
#include "CSSParserTokenRange.h"
#include "CSSPrimitiveValue.h"
#include "CSSPropertyParserConsumer+Ident.h"
#include "CSSPropertyParserConsumer+Length.h"
#include "CSSPropertyParsing.h"
#include "CSSValueKeywords.h"
#include "CSSValuePair.h"

namespace WebCore {
namespace CSSPropertyParserHelpers {

RefPtr<CSSValue> consumeSize(CSSParserTokenRange& range, const CSSParserContext& context)
{
    // <'size'> = <length [0,âˆž]>{1,2} | auto | [ <page-size> || [ portrait | landscape ] ]
    // https://drafts.csswg.org/css-page/#descdef-page-size

    if (consumeIdentRaw<CSSValueAuto>(range))
        return CSSPrimitiveValue::create(CSSValueAuto);

    if (auto width = consumeLength(range, context, ValueRange::NonNegative)) {
        auto height = consumeLength(range, context, ValueRange::NonNegative);
        if (!height)
            return width;
        return CSSValuePair::create(width.releaseNonNull(), height.releaseNonNull());
    }

    auto pageSize = CSSPropertyParsing::consumePageSize(range);
    auto orientation = consumeIdent<CSSValuePortrait, CSSValueLandscape>(range);
    if (!pageSize)
        pageSize = CSSPropertyParsing::consumePageSize(range);
    if (!orientation && !pageSize)
        return nullptr;
    if (pageSize && !orientation)
        return pageSize;
    if (!pageSize)
        return orientation;
    return CSSValuePair::create(pageSize.releaseNonNull(), orientation.releaseNonNull());
}

} // namespace CSSPropertyParserHelpers
} // namespace WebCore
