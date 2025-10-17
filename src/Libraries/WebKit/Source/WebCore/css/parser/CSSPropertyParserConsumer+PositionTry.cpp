/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 27, 2023.
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
#include "CSSPropertyParserConsumer+PositionTry.h"

#include "CSSParserContext.h"
#include "CSSParserTokenRange.h"
#include "CSSPropertyParserConsumer+Ident.h"
#include "CSSPropertyParserConsumer+List.h"
#include "CSSValueList.h"

namespace WebCore {
namespace CSSPropertyParserHelpers {

RefPtr<CSSValue> consumePositionTryFallbacks(CSSParserTokenRange& range, const CSSParserContext&)
{
    if (auto result = consumeIdent<CSSValueNone>(range))
        return result;

    auto consume = [](CSSParserTokenRange& range) -> RefPtr<CSSValue> {
        Vector<CSSValueID, 3> idents;
        while (auto ident = consumeIdentRaw<CSSValueFlipBlock, CSSValueFlipInline, CSSValueFlipStart>(range)) {
            if (idents.contains(*ident))
                return nullptr;
            idents.append(*ident);
        }
        CSSValueListBuilder list;
        for (auto ident : idents)
            list.append(CSSPrimitiveValue::create(ident));
        return CSSValueList::createSpaceSeparated(WTFMove(list));
    };
    return consumeCommaSeparatedListWithSingleValueOptimization(range, consume);
}

} // namespace CSSPropertyParserHelpers
} // namespace WebCore

