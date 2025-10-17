/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 29, 2025.
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
#include "CSSPropertyParserConsumer+Attr.h"

#include "CSSAttrValue.h"
#include "CSSParserContext.h"
#include "CSSParserTokenRange.h"
#include "CSSPrimitiveValue.h"
#include "CSSPropertyParserConsumer+Primitives.h"
#include <wtf/text/AtomString.h>

namespace WebCore {
namespace CSSPropertyParserHelpers {

RefPtr<CSSValue> consumeAttr(CSSParserTokenRange args, const CSSParserContext& context)
{
    // Standard says this should be:
    //
    // <attr()>    = attr( <attr-name> <attr-type>? , <declaration-value>?)
    // <attr-name> = [ <ident-token> '|' ]? <ident-token>
    // <attr-type> = type( <syntax> ) | string | <attr-unit>
    // https://drafts.csswg.org/css-values-5/#funcdef-attr

    // FIXME: Add support for complete <attr-name> syntax, including namespace support.
    // FIXME: Add support for <attr-type> syntax

    if (args.peek().type() != IdentToken)
        return nullptr;

    CSSParserToken token = args.consumeIncludingWhitespace();

    AtomString attrName;
    if (context.isHTMLDocument)
        attrName = token.value().convertToASCIILowercaseAtom();
    else
        attrName = token.value().toAtomString();

    if (!args.atEnd() && !consumeCommaIncludingWhitespace(args))
        return nullptr;

    RefPtr<CSSValue> fallback;
    if (args.peek().type() == StringToken) {
        token = args.consumeIncludingWhitespace();
        fallback = CSSPrimitiveValue::create(token.value().toString());
    }

    if (!args.atEnd())
        return nullptr;

    auto attr = CSSAttrValue::create(WTFMove(attrName), WTFMove(fallback));
    // FIXME: Consider moving to a CSSFunctionValue with a custom-ident rather than a special CSS_ATTR primitive value.

    return CSSPrimitiveValue::create(WTFMove(attr));
}

} // namespace CSSPropertyParserHelpers
} // namespace WebCore
