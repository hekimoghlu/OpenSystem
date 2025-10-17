/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 29, 2023.
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
#include "CSSPropertyParserConsumer+Integer.h"
#include "CSSPropertyParserConsumer+IntegerDefinitions.h"

#include "CSSCalcSymbolTable.h"
#include "CSSPrimitiveNumericTypes.h"
#include "CSSPropertyParserConsumer+CSSPrimitiveValueResolver.h"
#include "CSSPropertyParserConsumer+MetaConsumer.h"

namespace WebCore {
namespace CSSPropertyParserHelpers {

template<typename T> static RefPtr<CSSPrimitiveValue> consumeIntegerType(CSSParserTokenRange& range, const CSSParserContext& context)
{
    return CSSPrimitiveValueResolver<T>::consumeAndResolve(range, context, { });
}

RefPtr<CSSPrimitiveValue> consumeInteger(CSSParserTokenRange& range, const CSSParserContext& context)
{
    return consumeIntegerType<CSS::Integer<CSS::All, int>>(range, context);
}

RefPtr<CSSPrimitiveValue> consumeNonNegativeInteger(CSSParserTokenRange& range, const CSSParserContext& context)
{
    return consumeIntegerType<CSS::Integer<CSS::Range{0, CSS::Range::infinity}, int>>(range, context);
}

RefPtr<CSSPrimitiveValue> consumePositiveInteger(CSSParserTokenRange& range, const CSSParserContext& context)
{
    return consumeIntegerType<CSS::Integer<CSS::Range{1, CSS::Range::infinity}, unsigned>>(range, context);
}

RefPtr<CSSPrimitiveValue> consumeInteger(CSSParserTokenRange& range, const CSSParserContext& context, const CSS::Range& valueRange)
{
    if (valueRange == CSS::All)
        return consumeInteger(range, context);
    if (valueRange == CSS::Range{0, CSS::Range::infinity})
        return consumeNonNegativeInteger(range, context);
    if (valueRange == CSS::Range{1, CSS::Range::infinity})
        return consumePositiveInteger(range, context);

    RELEASE_ASSERT_NOT_REACHED();
}

} // namespace CSSPropertyParserHelpers
} // namespace WebCore
