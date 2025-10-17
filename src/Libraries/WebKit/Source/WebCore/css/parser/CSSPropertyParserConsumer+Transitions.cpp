/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 14, 2022.
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
#include "CSSPropertyParserConsumer+Transitions.h"

#include "CSSCalcSymbolTable.h"
#include "CSSParserContext.h"
#include "CSSParserTokenRange.h"
#include "CSSPrimitiveValue.h"
#include "CSSPropertyParserConsumer+Ident.h"
#include "CSSValueList.h"

namespace WebCore {
namespace CSSPropertyParserHelpers {

static RefPtr<CSSValue> consumeSingleTransitionPropertyIdent(CSSParserTokenRange& range, const CSSParserToken& token)
{
    if (token.id() == CSSValueAll)
        return consumeIdent(range);
    if (auto property = token.parseAsCSSPropertyID()) {
        range.consumeIncludingWhitespace();
        return CSSPrimitiveValue::create(property);
    }
    return consumeCustomIdent(range);
}

RefPtr<CSSValue> consumeSingleTransitionPropertyOrNone(CSSParserTokenRange& range, const CSSParserContext&)
{
    // This variant of consumeSingleTransitionProperty is used for the slightly different
    // parse rules used for the 'transition' shorthand which allows 'none':
    //
    // <single-transition-or-none> = [ none | <single-transition-property> ]
    // https://drafts.csswg.org/css-transitions/#single-transition-property

    auto& token = range.peek();
    if (token.type() != IdentToken)
        return nullptr;
    if (token.id() == CSSValueNone)
        return consumeIdent(range);

    return consumeSingleTransitionPropertyIdent(range, token);
}

RefPtr<CSSValue> consumeSingleTransitionProperty(CSSParserTokenRange& range, const CSSParserContext&)
{
    // "The <custom-ident> production in <single-transition-property> also excludes the keyword
    //  none, in addition to the keywords always excluded from <custom-ident>."
    //
    // <single-transition-property> = all | <custom-ident>;
    // https://drafts.csswg.org/css-transitions/#single-transition-property

    auto& token = range.peek();
    if (token.type() != IdentToken)
        return nullptr;
    if (token.id() == CSSValueNone)
        return nullptr;

    return consumeSingleTransitionPropertyIdent(range, token);
}

} // namespace CSSPropertyParserHelpers
} // namespace WebCore
