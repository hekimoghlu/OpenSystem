/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 3, 2024.
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
#pragma once

#include "CSSParserToken.h"
#include "CSSParserTokenRange.h"
#include "CSSPrimitiveValue.h"
#include "CSSValuePool.h"
#include <optional>
#include <wtf/RefPtr.h>

namespace WebCore {
namespace CSSPropertyParserHelpers {

// MARK: - Ident

std::optional<CSSValueID> consumeIdentRaw(CSSParserTokenRange&);
RefPtr<CSSPrimitiveValue> consumeIdent(CSSParserTokenRange&);
RefPtr<CSSPrimitiveValue> consumeIdentRange(CSSParserTokenRange&, CSSValueID lower, CSSValueID upper);

template<typename... emptyBaseCase> bool identMatches(CSSValueID)
{
    return false;
}

template<CSSValueID head, CSSValueID... tail> bool identMatches(CSSValueID id)
{
    return id == head || identMatches<tail...>(id);
}

template<CSSValueID... names> std::optional<CSSValueID> consumeIdentRaw(CSSParserTokenRange& range)
{
    if (range.peek().type() != IdentToken || !identMatches<names...>(range.peek().id()))
        return std::nullopt;
    return range.consumeIncludingWhitespace().id();
}

template<CSSValueID... names> RefPtr<CSSPrimitiveValue> consumeIdent(CSSParserTokenRange& range)
{
    if (range.peek().type() != IdentToken || !identMatches<names...>(range.peek().id()))
        return nullptr;
    return CSSPrimitiveValue::create(range.consumeIncludingWhitespace().id());
}

template<typename Predicate, typename... Args> std::optional<CSSValueID> consumeIdentRaw(CSSParserTokenRange& range, Predicate&& predicate, Args&&... args)
{
    if (auto keyword = range.peek().id(); predicate(keyword, std::forward<Args>(args)...)) {
        range.consumeIncludingWhitespace();
        return keyword;
    }
    return std::nullopt;
}

template<typename Predicate, typename... Args> RefPtr<CSSPrimitiveValue> consumeIdent(CSSParserTokenRange& range, Predicate&& predicate, Args&&... args)
{
    if (auto keyword = range.peek().id(); predicate(keyword, std::forward<Args>(args)...)) {
        range.consumeIncludingWhitespace();
        return CSSPrimitiveValue::create(keyword);
    }
    return nullptr;
}

template<typename Map>
std::optional<typename Map::ValueType> consumeIdentUsingMapping(CSSParserTokenRange& range, Map& map)
{
    if (auto value = map.tryGet(range.peek().id())) {
        range.consumeIncludingWhitespace();
        return std::make_optional(*value);
    }
    return std::nullopt;
}

template<typename Map>
std::optional<typename Map::ValueType> peekIdentUsingMapping(CSSParserTokenRange& range, Map& map)
{
    if (auto value = map.tryGet(range.peek().id()))
        return std::make_optional(*value);
    return std::nullopt;
}

// MARK: <custom-ident>
// https://drafts.csswg.org/css-values/#custom-idents

String consumeCustomIdentRaw(CSSParserTokenRange&, bool shouldLowercase = false);
RefPtr<CSSPrimitiveValue> consumeCustomIdent(CSSParserTokenRange&, bool shouldLowercase = false);

// MARK: <dashed-ident>
// https://drafts.csswg.org/css-values/#dashed-idents

String consumeDashedIdentRaw(CSSParserTokenRange&, bool shouldLowercase = false);
RefPtr<CSSPrimitiveValue> consumeDashedIdent(CSSParserTokenRange&, bool shouldLowercase = false);

} // namespace CSSPropertyParserHelpers
} // namespace WebCore
