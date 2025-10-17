/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 26, 2022.
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
#include "CSSPropertyParserConsumer+Conditional.h"

#include "CSSParserContext.h"
#include "CSSParserTokenRange.h"
#include "CSSPropertyParserConsumer+Ident.h"
#include "CSSValueList.h"

namespace WebCore {
namespace CSSPropertyParserHelpers {

RefPtr<CSSPrimitiveValue> consumeSingleContainerName(CSSParserTokenRange& range, const CSSParserContext&)
{
    // <single-container-name> = <custom-ident excluding=[none,and,or,not]>+
    // https://drafts.csswg.org/css-conditional-5/#propdef-container-name

    if (!isValidContainerNameIdentifier(range.peek().id()))
        return nullptr;
    return consumeCustomIdent(range);
}

RefPtr<CSSValue> consumeContainerName(CSSParserTokenRange& range, const CSSParserContext& context)
{
    // <'container-name'> = none | <custom-ident excluding=[none,and,or,not]>+
    // https://drafts.csswg.org/css-conditional-5/#propdef-container-name

    if (range.peek().id() == CSSValueNone)
        return consumeIdent(range);
    CSSValueListBuilder list;
    do {
        auto name = consumeSingleContainerName(range, context);
        if (!name)
            break;
        list.append(name.releaseNonNull());
    } while (!range.atEnd());
    if (list.isEmpty())
        return nullptr;
    return CSSValueList::createSpaceSeparated(WTFMove(list));
}

} // namespace CSSPropertyParserHelpers
} // namespace WebCore
