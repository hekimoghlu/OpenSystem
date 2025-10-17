/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 18, 2025.
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
#include "CSSPropertyParserConsumer+MetaConsumerDefinitions.h"
#include "CSSSymbol.h"
#include <optional>

namespace WebCore {

class CSSCalcSymbolsAllowed;
class CSSParserTokenRange;

struct CSSParserContext;
struct CSSPropertyParserOptions;

namespace CSSPropertyParserHelpers {

std::optional<CSS::SymbolRaw> validatedRange(CSS::SymbolRaw, CSSPropertyParserOptions);

struct SymbolKnownTokenTypeIdentConsumer {
    static constexpr CSSParserTokenType tokenType = IdentToken;
    static std::optional<CSS::SymbolRaw> consume(CSSParserTokenRange&, const CSSParserContext&, CSSCalcSymbolsAllowed, CSSPropertyParserOptions);
};

template<> struct ConsumerDefinition<CSS::Symbol> {
    using IdentToken = SymbolKnownTokenTypeIdentConsumer;
};

} // namespace CSSPropertyParserHelpers
} // namespace WebCore
