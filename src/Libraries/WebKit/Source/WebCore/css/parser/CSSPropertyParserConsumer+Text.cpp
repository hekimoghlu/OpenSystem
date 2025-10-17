/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 4, 2024.
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
#include "CSSPropertyParserConsumer+Text.h"

#include "CSSCalcSymbolTable.h"
#include "CSSParserContext.h"
#include "CSSParserTokenRange.h"
#include "CSSPrimitiveValue.h"
#include "CSSPropertyParserConsumer+Ident.h"
#include "CSSPropertyParserConsumer+LengthPercentage.h"
#include "CSSValueKeywords.h"
#include "CSSValueList.h"

namespace WebCore {
namespace CSSPropertyParserHelpers {

RefPtr<CSSValue> consumeTextIndent(CSSParserTokenRange& range, const CSSParserContext& context)
{
    // <'text-indent'> = [ <length-percentage> ] && hanging? && each-line?
    // https://drafts.csswg.org/css-text-3/#text-indent-property

    RefPtr<CSSValue> lengthPercentage;
    bool eachLine = false;
    bool hanging = false;
    do {
        if (!lengthPercentage) {
            if (auto textIndent = consumeLengthPercentage(range, context, ValueRange::All, UnitlessQuirk::Allow)) {
                lengthPercentage = textIndent;
                continue;
            }
        }
        if (!eachLine && consumeIdentRaw<CSSValueEachLine>(range)) {
            eachLine = true;
            continue;
        }
        if (!hanging && consumeIdentRaw<CSSValueHanging>(range)) {
            hanging = true;
            continue;
        }
        return nullptr;
    } while (!range.atEnd());
    if (!lengthPercentage)
        return nullptr;

    if (!hanging && !eachLine)
        return CSSValueList::createSpaceSeparated(lengthPercentage.releaseNonNull());
    if (hanging && !eachLine)
        return CSSValueList::createSpaceSeparated(lengthPercentage.releaseNonNull(), CSSPrimitiveValue::create(CSSValueHanging));
    if (!hanging)
        return CSSValueList::createSpaceSeparated(lengthPercentage.releaseNonNull(), CSSPrimitiveValue::create(CSSValueEachLine));
    return CSSValueList::createSpaceSeparated(lengthPercentage.releaseNonNull(),
        CSSPrimitiveValue::create(CSSValueHanging), CSSPrimitiveValue::create(CSSValueEachLine));
}

RefPtr<CSSValue> consumeTextTransform(CSSParserTokenRange& range, const CSSParserContext&)
{
    // <'text-transform'> = none | [capitalize | uppercase | lowercase ] || full-width || full-size-kana
    // https://drafts.csswg.org/css-text-3/#text-transform-property

    if (consumeIdentRaw<CSSValueNone>(range))
        return CSSPrimitiveValue::create(CSSValueNone);

    bool fullSizeKana = false;
    bool fullWidth = false;
    bool uppercase = false;
    bool capitalize = false;
    bool lowercase = false;

    do {
        auto ident = consumeIdentRaw(range);
        if (!ident)
            return nullptr;

        if (ident == CSSValueFullSizeKana && !fullSizeKana) {
            fullSizeKana = true;
            continue;
        }
        if (ident == CSSValueFullWidth && !fullWidth) {
            fullWidth = true;
            continue;
        }
        bool alreadySet = uppercase || capitalize || lowercase;
        if (ident == CSSValueUppercase && !alreadySet) {
            uppercase = true;
            continue;
        }
        if (ident == CSSValueCapitalize && !alreadySet) {
            capitalize = true;
            continue;
        }
        if (ident == CSSValueLowercase && !alreadySet) {
            lowercase = true;
            continue;
        }
        return nullptr;
    } while (!range.atEnd());

    // Construct the result list in canonical order
    CSSValueListBuilder list;
    if (capitalize)
        list.append(CSSPrimitiveValue::create(CSSValueCapitalize));
    else if (uppercase)
        list.append(CSSPrimitiveValue::create(CSSValueUppercase));
    else if (lowercase)
        list.append(CSSPrimitiveValue::create(CSSValueLowercase));

    if (fullWidth)
        list.append(CSSPrimitiveValue::create(CSSValueFullWidth));

    if (fullSizeKana)
        list.append(CSSPrimitiveValue::create(CSSValueFullSizeKana));

    return CSSValueList::createSpaceSeparated(WTFMove(list));
}

RefPtr<CSSValue> consumeHangingPunctuation(CSSParserTokenRange& range, const CSSParserContext&)
{
    // <'hanging-punctuation'> = none | [ first || [ force-end | allow-end ] || last ]
    // https://drafts.csswg.org/css-text-3/#propdef-hanging-punctuation

    if (range.peek().id() == CSSValueNone)
        return consumeIdent(range);

    CSSValueListBuilder list;

    bool seenForceEnd = false;
    bool seenAllowEnd = false;
    bool seenFirst = false;
    bool seenLast = false;

    while (!range.atEnd()) {
        CSSValueID valueID = range.peek().id();
        if ((valueID == CSSValueFirst && seenFirst)
            || (valueID == CSSValueLast && seenLast)
            || (valueID == CSSValueAllowEnd && (seenAllowEnd || seenForceEnd))
            || (valueID == CSSValueForceEnd && (seenAllowEnd || seenForceEnd)))
            return nullptr;
        auto ident = consumeIdent<CSSValueAllowEnd, CSSValueForceEnd, CSSValueFirst, CSSValueLast>(range);
        if (!ident)
            return nullptr;
        switch (valueID) {
        case CSSValueAllowEnd:
            seenAllowEnd = true;
            break;
        case CSSValueForceEnd:
            seenForceEnd = true;
            break;
        case CSSValueFirst:
            seenFirst = true;
            break;
        case CSSValueLast:
            seenLast = true;
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

RefPtr<CSSValue> consumeTextAutospace(CSSParserTokenRange& range, const CSSParserContext&)
{
    // <'text-autospace'> = normal | <autospace> | auto
    // <autospace>        = no-autospace | [ ideograph-alpha || ideograph-numeric || punctuation ] || [ insert | replace ]
    // https://drafts.csswg.org/css-text-4/#text-autospace-property

    // FIXME: add remaining values;
    if (auto value = consumeIdent<CSSValueAuto, CSSValueNoAutospace, CSSValueNormal>(range)) {
        if (!range.atEnd())
            return nullptr;
        return value;
    }

    CSSValueListBuilder list;
    bool seenIdeographAlpha = false;
    bool seenIdeographNumeric = false;

    while (!range.atEnd()) {
        auto valueID = range.peek().id();

        if ((valueID == CSSValueIdeographAlpha && seenIdeographAlpha) || (valueID == CSSValueIdeographNumeric && seenIdeographNumeric))
            return nullptr;

        auto ident = consumeIdent<CSSValueIdeographAlpha, CSSValueIdeographNumeric>(range);
        if (!ident)
            return nullptr;
        switch (valueID) {
        case CSSValueIdeographAlpha:
            seenIdeographAlpha = true;
            break;
        case CSSValueIdeographNumeric:
            seenIdeographNumeric = true;
            break;
        default:
            return nullptr;
        }
    }

    if (seenIdeographAlpha)
        list.append(CSSPrimitiveValue::create(CSSValueIdeographAlpha));
    if (seenIdeographNumeric)
        list.append(CSSPrimitiveValue::create(CSSValueIdeographNumeric));

    if (list.isEmpty())
        return nullptr;
    return CSSValueList::createSpaceSeparated(WTFMove(list));
}

} // namespace CSSPropertyParserHelpers
} // namespace WebCore
