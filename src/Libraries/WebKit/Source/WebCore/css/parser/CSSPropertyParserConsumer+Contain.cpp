/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 1, 2022.
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
#include "CSSPropertyParserConsumer+Contain.h"

#include "CSSParserTokenRange.h"
#include "CSSPrimitiveValue.h"
#include "CSSPropertyParserConsumer+Ident.h"
#include "CSSValueKeywords.h"
#include "CSSValueList.h"

namespace WebCore {
namespace CSSPropertyParserHelpers {

RefPtr<CSSValue> consumeContain(CSSParserTokenRange& range, const CSSParserContext&)
{
    // <'contain'> = none | strict | content | [ [size | inline-size] || layout || style || paint ]
    // https://drafts.csswg.org/css-contain-2/#propdef-contain

    if (auto singleValue = consumeIdent<CSSValueNone, CSSValueStrict, CSSValueContent>(range))
        return singleValue;

    RefPtr<CSSPrimitiveValue> size;
    RefPtr<CSSPrimitiveValue> inlineSize;
    RefPtr<CSSPrimitiveValue> layout;
    RefPtr<CSSPrimitiveValue> style;
    RefPtr<CSSPrimitiveValue> paint;

    while (!range.atEnd()) {
        switch (range.peek().id()) {
        case CSSValueSize:
            if (size)
                return nullptr;
            size = consumeIdent(range);
            break;
        case CSSValueInlineSize:
            if (inlineSize || size)
                return nullptr;
            inlineSize = consumeIdent(range);
            break;
        case CSSValueLayout:
            if (layout)
                return nullptr;
            layout = consumeIdent(range);
            break;
        case CSSValuePaint:
            if (paint)
                return nullptr;
            paint = consumeIdent(range);
            break;
        case CSSValueStyle:
            if (style)
                return nullptr;
            style = consumeIdent(range);
            break;
        default:
            return nullptr;
        }
    }

    CSSValueListBuilder list;
    if (size)
        list.append(size.releaseNonNull());
    if (inlineSize)
        list.append(inlineSize.releaseNonNull());
    if (layout)
        list.append(layout.releaseNonNull());
    if (style)
        list.append(style.releaseNonNull());
    if (paint)
        list.append(paint.releaseNonNull());

    if (list.isEmpty())
        return nullptr;
    return CSSValueList::createSpaceSeparated(WTFMove(list));
}

} // namespace CSSPropertyParserHelpers
} // namespace WebCore
