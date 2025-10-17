/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 8, 2025.
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
#include "CSSPropertyParserConsumer+WillChange.h"

#include "CSSParserContext.h"
#include "CSSParserTokenRange.h"
#include "CSSPrimitiveValue.h"
#include "CSSPropertyNames.h"
#include "CSSPropertyParser.h"
#include "CSSPropertyParserConsumer+Ident.h"
#include "CSSPropertyParserConsumer+Primitives.h"
#include "CSSValueKeywords.h"
#include "CSSValueList.h"

namespace WebCore {
namespace CSSPropertyParserHelpers {

RefPtr<CSSValue> consumeWillChange(CSSParserTokenRange& range, const CSSParserContext& context)
{
    // <'will-change'> = auto | <animateable-feature>#
    // https://drafts.csswg.org/css-will-change/#propdef-will-change

    if (range.peek().id() == CSSValueAuto)
        return consumeIdent(range);

    // Every comma-separated list of identifiers is a valid will-change value,
    // unless the list includes an explicitly disallowed identifier.
    CSSValueListBuilder values;
    while (!range.atEnd()) {
        switch (range.peek().id()) {
        case CSSValueContents:
        case CSSValueScrollPosition:
            values.append(consumeIdent(range).releaseNonNull());
            break;
        case CSSValueNone:
        case CSSValueAll:
        case CSSValueAuto:
            return nullptr;
        default:
            if (range.peek().type() != IdentToken)
                return nullptr;
            CSSPropertyID propertyID = cssPropertyID(range.peek().value());
            if (propertyID == CSSPropertyWillChange)
                return nullptr;
            if (!isExposed(propertyID, &context.propertySettings))
                propertyID = CSSPropertyInvalid;
            if (propertyID != CSSPropertyInvalid) {
                values.append(CSSPrimitiveValue::create(propertyID));
                range.consumeIncludingWhitespace();
                break;
            }
            if (auto customIdent = consumeCustomIdent(range)) {
                // Append properties we don't recognize, but that are legal.
                values.append(customIdent.releaseNonNull());
                break;
            }
            return nullptr;
        }
        // This is a comma separated list
        if (!range.atEnd() && !consumeCommaIncludingWhitespace(range))
            return nullptr;
    }
    return CSSValueList::createCommaSeparated(WTFMove(values));
}

} // namespace CSSPropertyParserHelpers
} // namespace WebCore
