/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 10, 2025.
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
#include "CSSPropertyParserConsumer+Speech.h"

#include "CSSParserContext.h"
#include "CSSParserTokenRange.h"
#include "CSSPrimitiveValue.h"
#include "CSSPropertyParserConsumer+Ident.h"
#include "CSSValueKeywords.h"
#include "CSSValueList.h"

namespace WebCore {
namespace CSSPropertyParserHelpers {

RefPtr<CSSValue> consumeSpeakAs(CSSParserTokenRange& range, const CSSParserContext&)
{
    // <'speak-as'> = normal | spell-out || digits || [ literal-punctuation | no-punctuation ]
    // https://drafts.csswg.org/css-speech-1/#propdef-speak-as

    if (range.peek().id() == CSSValueNone)
        return consumeIdent(range);

    CSSValueListBuilder list;

    bool seenNormal = false;
    bool seenSpellOut = false;
    bool seenLiteralPunctuation = false;
    bool seenNoPunctuation = false;

    while (!range.atEnd()) {
        CSSValueID valueID = range.peek().id();
        if ((valueID == CSSValueNormal && seenSpellOut)
            || (valueID == CSSValueSpellOut && seenNormal)
            || (valueID == CSSValueLiteralPunctuation && seenNoPunctuation)
            || (valueID == CSSValueNoPunctuation && seenLiteralPunctuation))
            return nullptr;
        auto ident = consumeIdent<CSSValueNormal, CSSValueSpellOut, CSSValueDigits, CSSValueLiteralPunctuation, CSSValueNoPunctuation>(range);
        if (!ident)
            return nullptr;
        switch (valueID) {
        case CSSValueNormal:
            seenNormal = true;
            break;
        case CSSValueSpellOut:
            seenSpellOut = true;
            break;
        case CSSValueLiteralPunctuation:
            seenLiteralPunctuation = true;
            break;
        case CSSValueNoPunctuation:
            seenNoPunctuation = true;
            break;
        default:
            break;
        }
        list.append(ident.releaseNonNull());
    }

    if (list.isEmpty())
        return nullptr;
    return CSSValueList::createSpaceSeparated(WTFMove(list));
}

} // namespace CSSPropertyParserHelpers
} // namespace WebCore
