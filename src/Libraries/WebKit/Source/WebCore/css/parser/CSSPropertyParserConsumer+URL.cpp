/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 20, 2023.
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
#include "CSSPropertyParserConsumer+URL.h"

#include "CSSParserTokenRange.h"
#include "CSSPrimitiveValue.h"
#include "CSSValueKeywords.h"
#include <wtf/text/StringView.h>

namespace WebCore {
namespace CSSPropertyParserHelpers {

// MARK: <url>
// https://drafts.csswg.org/css-values/#urls

StringView consumeURLRaw(CSSParserTokenRange& range)
{
    auto& token = range.peek();
    if (token.type() == UrlToken) {
        range.consumeIncludingWhitespace();
        return token.value();
    }

    if (token.functionId() == CSSValueUrl) {
        auto rangeCopy = range;
        auto args = rangeCopy.consumeBlock();
        auto& next = args.consumeIncludingWhitespace();
        if (next.type() == BadStringToken || !args.atEnd())
            return StringView();
        ASSERT(next.type() == StringToken);
        range = rangeCopy;
        range.consumeWhitespace();
        return next.value();
    }

    return { };
}

RefPtr<CSSPrimitiveValue> consumeURL(CSSParserTokenRange& range)
{
    auto url = consumeURLRaw(range);
    if (url.isNull())
        return nullptr;
    return CSSPrimitiveValue::createURI(url.toString());
}

} // namespace CSSPropertyParserHelpers
} // namespace WebCore
