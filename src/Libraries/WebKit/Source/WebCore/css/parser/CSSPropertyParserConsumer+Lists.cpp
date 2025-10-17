/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 28, 2024.
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
#include "CSSPropertyParserConsumer+Lists.h"

#include "CSSParserContext.h"
#include "CSSParserTokenRange.h"
#include "CSSPrimitiveValue.h"
#include "CSSPropertyParserConsumer+CounterStyles.h"
#include "CSSPropertyParserConsumer+Ident.h"
#include "CSSPropertyParserConsumer+Integer.h"
#include "CSSPropertyParserConsumer+String.h"
#include "CSSValueKeywords.h"
#include "CSSValueList.h"
#include "CSSValuePair.h"

namespace WebCore {
namespace CSSPropertyParserHelpers {

static RefPtr<CSSValue> consumeCounter(CSSParserTokenRange& range, const CSSParserContext& context, int defaultValue)
{
    if (range.peek().id() == CSSValueNone)
        return consumeIdent(range);

    CSSValueListBuilder list;
    do {
        auto counterName = consumeCustomIdent(range);
        if (!counterName)
            return nullptr;
        if (auto counterValue = consumeInteger(range, context))
            list.append(CSSValuePair::create(counterName.releaseNonNull(), counterValue.releaseNonNull()));
        else
            list.append(CSSValuePair::create(counterName.releaseNonNull(), CSSPrimitiveValue::createInteger(defaultValue)));
    } while (!range.atEnd());
    return CSSValueList::createSpaceSeparated(WTFMove(list));
}

RefPtr<CSSValue> consumeCounterReset(CSSParserTokenRange& range, const CSSParserContext& context)
{
    // <'counter-reset'> = [ <counter-name> <integer>? | <reversed-counter-name> <integer>? ]+ | none
    // https://drafts.csswg.org/css-lists/#propdef-counter-reset

    // FIXME: Implement support for `reversed-counter-name`.

    return consumeCounter(range, context, 0);
}

RefPtr<CSSValue> consumeCounterIncrement(CSSParserTokenRange& range, const CSSParserContext& context)
{
    // <'counter-increment'> = [ <counter-name> <integer>? ]+ | none
    // https://drafts.csswg.org/css-lists/#propdef-counter-increment

    return consumeCounter(range, context, 1);
}

RefPtr<CSSValue> consumeCounterSet(CSSParserTokenRange& range, const CSSParserContext& context)
{
    // <'counter-set'> = [ <counter-name> <integer>? ]+ | none
    // https://drafts.csswg.org/css-lists/#propdef-counter-set

    return consumeCounter(range, context, 0);
}

RefPtr<CSSValue> consumeListStyleType(CSSParserTokenRange& range, const CSSParserContext& context)
{
    // <'list-style-type'> = <counter-style> | <string> | none
    // https://drafts.csswg.org/css-lists/#propdef-list-style-type

    if (range.peek().id() == CSSValueNone)
        return consumeIdent(range);
    if (range.peek().type() == StringToken)
        return consumeString(range);
    return consumeCounterStyle(range, context);
}

} // namespace CSSPropertyParserHelpers
} // namespace WebCore
