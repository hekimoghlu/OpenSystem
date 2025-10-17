/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 10, 2025.
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

#include "CSSParserMode.h"
#include <wtf/Forward.h>

namespace WebCore {

class CSSParserTokenRange;
class CSSValue;
struct CSSParserContext;
enum CSSValueID : uint16_t;

namespace CSSPropertyParserHelpers {

// https://drafts.csswg.org/css-counter-styles-3/

// MARK: <counter-style> consumer
RefPtr<CSSValue> consumeCounterStyle(CSSParserTokenRange&, const CSSParserContext&);

// MARK: @counter-style descriptor consumers
AtomString consumeCounterStyleNameInPrelude(CSSParserTokenRange&, CSSParserMode = CSSParserMode::HTMLStandardMode);
RefPtr<CSSValue> consumeCounterStyleName(CSSParserTokenRange&, const CSSParserContext&);
RefPtr<CSSValue> consumeCounterStyleSymbol(CSSParserTokenRange&, const CSSParserContext&);
RefPtr<CSSValue> consumeCounterStyleSystem(CSSParserTokenRange&, const CSSParserContext&);
RefPtr<CSSValue> consumeCounterStyleNegative(CSSParserTokenRange&, const CSSParserContext&);
RefPtr<CSSValue> consumeCounterStyleRange(CSSParserTokenRange&, const CSSParserContext&);
RefPtr<CSSValue> consumeCounterStylePad(CSSParserTokenRange&, const CSSParserContext&);
RefPtr<CSSValue> consumeCounterStyleSymbols(CSSParserTokenRange&, const CSSParserContext&);
RefPtr<CSSValue> consumeCounterStyleAdditiveSymbols(CSSParserTokenRange&, const CSSParserContext&);
RefPtr<CSSValue> consumeCounterStyleSpeakAs(CSSParserTokenRange&, const CSSParserContext&);

} // namespace CSSPropertyParserHelpers
} // namespace WebCore
