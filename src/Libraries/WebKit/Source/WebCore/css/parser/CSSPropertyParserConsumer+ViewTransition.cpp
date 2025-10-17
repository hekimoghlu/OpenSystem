/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 18, 2024.
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
#include "CSSPropertyParserConsumer+ViewTransition.h"

#include "CSSParserTokenRange.h"
#include "CSSPrimitiveValue.h"
#include "CSSPropertyParserConsumer+Ident.h"
#include "CSSValueKeywords.h"
#include "CSSValueList.h"
#include "CSSValuePool.h"

namespace WebCore {
namespace CSSPropertyParserHelpers {

RefPtr<CSSValue> consumeViewTransitionClass(CSSParserTokenRange& range, const CSSParserContext&)
{
    // <'view-transition-class'> = none | <custom-ident>+
    // https://drafts.csswg.org/css-view-transitions-2/#view-transition-class-prop

    if (auto noneValue = consumeIdent<CSSValueNone>(range))
        return noneValue;

    CSSValueListBuilder list;
    do {
        if (range.peek().id() == CSSValueNone)
            return nullptr;

        auto ident = consumeCustomIdent(range);
        if (!ident)
            return nullptr;

        list.append(ident.releaseNonNull());
    } while (!range.atEnd());

    if (list.isEmpty())
        return nullptr;

    return CSSValueList::createSpaceSeparated(WTFMove(list));
}

RefPtr<CSSValue> consumeViewTransitionTypes(CSSParserTokenRange& range, const CSSParserContext&)
{
    // <'types'> = none | <custom-ident>+
    // https://www.w3.org/TR/css-view-transitions-2/#descdef-view-transition-types

    if (range.peek().id() == CSSValueNone)
        return consumeIdent(range);

    CSSValueListBuilder list;
    do {
        if (range.peek().id() == CSSValueNone)
            return nullptr;
        auto type = consumeCustomIdent(range);
        if (!type)
            return nullptr;
        if (type->customIdent().startsWith("-ua-"_s))
            return nullptr;
        list.append(type.releaseNonNull());
    } while (!range.atEnd());
    return CSSValueList::createSpaceSeparated(WTFMove(list));
}

} // namespace CSSPropertyParserHelpers
} // namespace WebCore
