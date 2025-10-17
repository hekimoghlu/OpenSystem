/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 29, 2022.
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
#include "CSSPropertyParserConsumer+Content.h"

#include "CSSCounterValue.h"
#include "CSSParserContext.h"
#include "CSSParserTokenRange.h"
#include "CSSPrimitiveValue.h"
#include "CSSPropertyParserConsumer+Attr.h"
#include "CSSPropertyParserConsumer+Ident.h"
#include "CSSPropertyParserConsumer+Image.h"
#include "CSSPropertyParserConsumer+Lists.h"
#include "CSSPropertyParserConsumer+Primitives.h"
#include "CSSPropertyParserConsumer+String.h"
#include "CSSValueKeywords.h"
#include "CSSValueList.h"
#include "CSSValuePair.h"

namespace WebCore {
namespace CSSPropertyParserHelpers {

RefPtr<CSSValue> consumeQuotes(CSSParserTokenRange& range, const CSSParserContext&)
{
    // <'quotes'> = auto | none | match-parent | [ <string> <string> ]+
    // https://drafts.csswg.org/css-content-3/#propdef-quotes

    // FIXME: Support `match-parent`.

    auto id = range.peek().id();
    if (id == CSSValueNone || id == CSSValueAuto)
        return consumeIdent(range);

    CSSValueListBuilder values;
    while (!range.atEnd()) {
        auto parsedValue = consumeString(range);
        if (!parsedValue)
            return nullptr;
        values.append(parsedValue.releaseNonNull());
    }
    if (values.size() && !(values.size() % 2))
        return CSSValueList::createSpaceSeparated(WTFMove(values));
    return nullptr;
}

static RefPtr<CSSValue> consumeCounterContent(CSSParserTokenRange args, const CSSParserContext& context, bool counters)
{
    AtomString identifier { consumeCustomIdentRaw(args) };
    if (identifier.isNull())
        return nullptr;

    AtomString separator;
    if (counters) {
        if (!consumeCommaIncludingWhitespace(args) || args.peek().type() != StringToken)
            return nullptr;
        separator = args.consumeIncludingWhitespace().value().toAtomString();
    }

    RefPtr<CSSValue> listStyleType = CSSPrimitiveValue::create(CSSValueDecimal);
    if (consumeCommaIncludingWhitespace(args)) {
        if (args.peek().id() == CSSValueNone || args.peek().type() == StringToken)
            return nullptr;
        listStyleType = consumeListStyleType(args, context);
        if (!listStyleType)
            return nullptr;
    }

    if (!args.atEnd())
        return nullptr;

    return CSSCounterValue::create(WTFMove(identifier), WTFMove(separator), WTFMove(listStyleType));
}

RefPtr<CSSValue> consumeContent(CSSParserTokenRange& range, const CSSParserContext& context)
{
    // Standard says this should be:
    //
    // <'content'> = normal | none | [ <content-replacement> | <content-list> ] [/ [ <string> | <counter> | <attr()> ]+ ]?
    // https://drafts.csswg.org/css-content-3/#propdef-content

    if (identMatches<CSSValueNone, CSSValueNormal>(range.peek().id()))
        return consumeIdent(range);

    enum class ContentListType : bool { VisibleContent, AltText };
    auto consumeContentList = [&](CSSValueListBuilder& values, ContentListType type) -> bool {
        bool shouldEnd = false;
        do {
            RefPtr<CSSValue> parsedValue = consumeString(range);
            if (type == ContentListType::VisibleContent) {
                if (!parsedValue)
                    parsedValue = consumeImage(range, context);
                if (!parsedValue)
                    parsedValue = consumeIdent<CSSValueOpenQuote, CSSValueCloseQuote, CSSValueNoOpenQuote, CSSValueNoCloseQuote>(range);
            }
            if (!parsedValue) {
                if (range.peek().functionId() == CSSValueAttr)
                    parsedValue = consumeAttr(consumeFunction(range), context);
                // FIXME: Alt-text should support counters.
                else if (type == ContentListType::VisibleContent) {
                    if (range.peek().functionId() == CSSValueCounter)
                        parsedValue = consumeCounterContent(consumeFunction(range), context, false);
                    else if (range.peek().functionId() == CSSValueCounters)
                        parsedValue = consumeCounterContent(consumeFunction(range), context, true);
                }
                if (!parsedValue)
                    return false;
            }
            values.append(parsedValue.releaseNonNull());

            // Visible content parsing ends at '/' or end of range.
            if (type == ContentListType::VisibleContent && !range.atEnd()) {
                CSSParserToken value = range.peek();
                if (value.type() == DelimiterToken && value.delimiter() == '/')
                    shouldEnd = true;
            }
            shouldEnd = shouldEnd || range.atEnd();
        } while (!shouldEnd);
        return true;
    };

    CSSValueListBuilder visibleContent;
    if (!consumeContentList(visibleContent, ContentListType::VisibleContent))
        return nullptr;

    // Consume alt-text content if there is any.
    if (consumeSlashIncludingWhitespace(range)) {
        CSSValueListBuilder altText;
        if (!consumeContentList(altText, ContentListType::AltText))
            return nullptr;
        return CSSValuePair::createSlashSeparated(
            CSSValueList::createSpaceSeparated(WTFMove(visibleContent)),
            CSSValueList::createSpaceSeparated(WTFMove(altText))
        );
    }

    return CSSValueList::createSpaceSeparated(WTFMove(visibleContent));
}


} // namespace CSSPropertyParserHelpers
} // namespace WebCore
