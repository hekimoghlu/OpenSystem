/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 10, 2022.
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
#include "CSSPropertyParserConsumer+Overflow.h"

#include "CSSParserTokenRange.h"
#include "CSSPrimitiveValue.h"
#include "CSSPropertyParserConsumer+Ident.h"
#include "CSSValueKeywords.h"
#include "CSSValuePair.h"

namespace WebCore {
namespace CSSPropertyParserHelpers {

RefPtr<CSSValue> consumeScrollbarGutter(CSSParserTokenRange& range, const CSSParserContext&)
{
    // <'scrollbar-gutter'> = auto | stable && both-edges?
    // https://drafts.csswg.org/css-overflow/#propdef-scrollbar-gutter

    if (auto ident = consumeIdent<CSSValueAuto>(range))
        return CSSPrimitiveValue::create(CSSValueAuto);

    if (auto first = consumeIdent<CSSValueStable>(range)) {
        if (auto second = consumeIdent<CSSValueBothEdges>(range))
            return CSSValuePair::create(first.releaseNonNull(), second.releaseNonNull());
        return first;
    }

    if (auto first = consumeIdent<CSSValueBothEdges>(range)) {
        if (auto second = consumeIdent<CSSValueStable>(range))
            return CSSValuePair::create(second.releaseNonNull(), first.releaseNonNull());
        return nullptr;
    }

    return nullptr;
}

} // namespace CSSPropertyParserHelpers
} // namespace WebCore
