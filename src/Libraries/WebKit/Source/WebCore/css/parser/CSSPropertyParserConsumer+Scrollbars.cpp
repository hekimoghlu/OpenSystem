/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 21, 2025.
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
#include "CSSPropertyParserConsumer+Scrollbars.h"

#include "CSSParserContext.h"
#include "CSSParserTokenRange.h"
#include "CSSPrimitiveValue.h"
#include "CSSPropertyParserConsumer+Color.h"
#include "CSSPropertyParserConsumer+Ident.h"
#include "CSSValueKeywords.h"
#include "CSSValuePair.h"

namespace WebCore {
namespace CSSPropertyParserHelpers {

RefPtr<CSSValue> consumeScrollbarColor(CSSParserTokenRange& range, const CSSParserContext& context)
{
    // <'scrollbar-color'> = auto | <color>{2}
    // https://drafts.csswg.org/css-scrollbars/#propdef-scrollbar-color

    if (auto ident = consumeIdent<CSSValueAuto>(range))
        return ident;

    if (auto thumbColor = consumeColor(range, context)) {
        if (auto trackColor = consumeColor(range, context))
            return CSSValuePair::createNoncoalescing(thumbColor.releaseNonNull(), trackColor.releaseNonNull());
    }

    return nullptr;
}

} // namespace CSSPropertyParserHelpers
} // namespace WebCore
