/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 22, 2024.
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
#include "CSSPropertyParserConsumer+ColorAdjust.h"

#include "CSSColorScheme.h"
#include "CSSColorSchemeValue.h"
#include "CSSParserContext.h"
#include "CSSParserIdioms.h"
#include "CSSParserTokenRange.h"
#include "CSSValueKeywords.h"

namespace WebCore {
namespace CSSPropertyParserHelpers {

#if ENABLE(DARK_MODE_CSS)

std::optional<CSS::ColorScheme> consumeUnresolvedColorScheme(CSSParserTokenRange& range, const CSSParserContext&)
{
    // <'color-scheme'> = normal | [ light | dark | <custom-ident> ]+ && only?
    // https://drafts.csswg.org/css-color-adjust/#propdef-color-scheme

    if (range.peek().id() == CSSValueNormal) {
        range.consumeIncludingWhitespace();

        // NOTE: `normal` is represented in CSS::ColorScheme as an empty list of schemes.
        return CSS::ColorScheme {
            .schemes = { },
            .only = { }
        };
    }

    std::optional<CSS::ColorScheme> result = CSS::ColorScheme {
        .schemes = { },
        .only = { }
    };

    if (range.peek().id() == CSSValueOnly) {
        range.consumeIncludingWhitespace();

        result->only = CSS::Keyword::Only { };
    }

    while (!range.atEnd()) {
        if (range.peek().type() != IdentToken)
            return { };

        CSSValueID id = range.peek().id();

        switch (id) {
        case CSSValueNormal:
            // `normal` is only allowed as a single value, and was handled earlier.
            // Don't allow it in the list.
            return { };

        case CSSValueOnly:
            // `only` can either appear first, handled before the loop, or last,
            // handled here.

            if (result->only)
                return { };
            range.consumeIncludingWhitespace();
            result->only = CSS::Keyword::Only { };

            if (!range.atEnd())
                return { };

            break;

        default:
            if (!isValidCustomIdentifier(id))
                return { };

            result->schemes.value.append(CustomIdentifier { range.consumeIncludingWhitespace().value().toAtomString() });
            break;
        }
    }

    if (result->schemes.isEmpty())
        return { };

    return result;
}

std::optional<CSS::ColorScheme> parseUnresolvedColorScheme(const String& string, const CSSParserContext& context)
{
    CSSTokenizer tokenizer(string);
    CSSParserTokenRange range(tokenizer.tokenRange());

    // Handle leading whitespace.
    range.consumeWhitespace();

    auto result = consumeUnresolvedColorScheme(range, context);

    // Handle trailing whitespace.
    range.consumeWhitespace();

    if (!range.atEnd())
        return { };

    return result;
}

RefPtr<CSSValue> consumeColorScheme(CSSParserTokenRange& range, const CSSParserContext& context)
{
    auto colorScheme = consumeUnresolvedColorScheme(range, context);
    if (!colorScheme)
        return { };
    return CSSColorSchemeValue::create(WTFMove(*colorScheme));
}

#endif // ENABLE(DARK_MODE_CSS)

} // namespace CSSPropertyParserHelpers
} // namespace WebCore
