/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 19, 2025.
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
#include "CSSPropertyParserConsumer+Ident.h"

#include "CSSParserIdioms.h"
#include "CSSValuePool.h"

namespace WebCore {
namespace CSSPropertyParserHelpers {

std::optional<CSSValueID> consumeIdentRaw(CSSParserTokenRange& range)
{
    if (range.peek().type() != IdentToken)
        return std::nullopt;
    return range.consumeIncludingWhitespace().id();
}

RefPtr<CSSPrimitiveValue> consumeIdent(CSSParserTokenRange& range)
{
    if (auto result = consumeIdentRaw(range))
        return CSSPrimitiveValue::create(*result);
    return nullptr;
}

static std::optional<CSSValueID> consumeIdentRangeRaw(CSSParserTokenRange& range, CSSValueID lower, CSSValueID upper)
{
    if (range.peek().id() < lower || range.peek().id() > upper)
        return std::nullopt;
    return consumeIdentRaw(range);
}

RefPtr<CSSPrimitiveValue> consumeIdentRange(CSSParserTokenRange& range, CSSValueID lower, CSSValueID upper)
{
    auto value = consumeIdentRangeRaw(range, lower, upper);
    if (!value)
        return nullptr;
    return CSSPrimitiveValue::create(*value);
}

// MARK: <custom-ident>
// https://drafts.csswg.org/css-values/#custom-idents

String consumeCustomIdentRaw(CSSParserTokenRange& range, bool shouldLowercase)
{
    if (range.peek().type() != IdentToken || !isValidCustomIdentifier(range.peek().id()))
        return String();
    auto identifier = range.consumeIncludingWhitespace().value();
    return shouldLowercase ? identifier.convertToASCIILowercase() : identifier.toString();
}

RefPtr<CSSPrimitiveValue> consumeCustomIdent(CSSParserTokenRange& range, bool shouldLowercase)
{
    auto identifier = consumeCustomIdentRaw(range, shouldLowercase);
    if (identifier.isNull())
        return nullptr;
    return CSSPrimitiveValue::createCustomIdent(WTFMove(identifier));
}

// MARK: <dashed-ident>
// https://drafts.csswg.org/css-values/#dashed-idents

String consumeDashedIdentRaw(CSSParserTokenRange& range, bool shouldLowercase)
{
    auto rangeCopy = range;
    auto identifier = consumeCustomIdentRaw(range, shouldLowercase);
    if (!identifier.startsWith("--"_s)) {
        range = rangeCopy;
        return { };
    }
    return identifier;
}

RefPtr<CSSPrimitiveValue> consumeDashedIdent(CSSParserTokenRange& range, bool shouldLowercase)
{
    auto identifier = consumeDashedIdentRaw(range, shouldLowercase);
    if (identifier.isNull())
        return nullptr;
    return CSSPrimitiveValue::createCustomIdent(WTFMove(identifier));
}

} // namespace CSSPropertyParserHelpers
} // namespace WebCore
