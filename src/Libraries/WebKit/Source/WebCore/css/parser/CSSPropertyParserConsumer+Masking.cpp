/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 28, 2023.
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
#include "CSSPropertyParserConsumer+Masking.h"

#include "CSSParserTokenRange.h"
#include "CSSPrimitiveValue.h"
#include "CSSPropertyParserConsumer+Ident.h"
#include "CSSPropertyParserConsumer+Length.h"
#include "CSSPropertyParserConsumer+Primitives.h"
#include "CSSPropertyParserConsumer+Shapes.h"
#include "CSSPropertyParserConsumer+URL.h"
#include "CSSPropertyParsing.h"
#include "CSSRectValue.h"
#include "CSSValueKeywords.h"
#include "CSSValueList.h"

namespace WebCore {
namespace CSSPropertyParserHelpers {

static RefPtr<CSSValue> consumeClipRectFunction(CSSParserTokenRange& range, const CSSParserContext& context)
{
    // rect() = rect( <top>, <right>, <bottom>, <left> )
    // "<top>, <right>, <bottom>, and <left> may either have a <length> value or auto."
    // https://drafts.fxtf.org/css-masking/#funcdef-clip-rect

    if (range.peek().functionId() != CSSValueRect)
        return nullptr;

    CSSParserTokenRange args = consumeFunction(range);

    auto consumeClipComponent = [&] -> RefPtr<CSSPrimitiveValue> {
        if (args.peek().id() == CSSValueAuto)
            return consumeIdent(args);
        return consumeLength(args, context, ValueRange::All, UnitlessQuirk::Allow);
    };

    // Support both rect(t, r, b, l) and rect(t r b l).
    //
    // "User agents must support separation with commas, but may also support
    //  separation without commas (but not a combination), because a previous
    //  revision of this specification was ambiguous in this respect"
    auto top = consumeClipComponent();
    if (!top)
        return nullptr;

    bool needsComma = consumeCommaIncludingWhitespace(args);

    auto right = consumeClipComponent();
    if (!right || (needsComma && !consumeCommaIncludingWhitespace(args)))
        return nullptr;

    auto bottom = consumeClipComponent();
    if (!bottom || (needsComma && !consumeCommaIncludingWhitespace(args)))
        return nullptr;

    auto left = consumeClipComponent();
    if (!left || !args.atEnd())
        return nullptr;

    return CSSRectValue::create(
        Rect {
            top.releaseNonNull(),
            right.releaseNonNull(),
            bottom.releaseNonNull(),
            left.releaseNonNull()
        }
    );
}

RefPtr<CSSValue> consumeClip(CSSParserTokenRange& range, const CSSParserContext& context)
{
    // <'clip'> = <rect()> | auto
    // https://drafts.fxtf.org/css-masking/#propdef-clip

    if (range.peek().id() == CSSValueAuto)
        return consumeIdent(range);
    return consumeClipRectFunction(range, context);
}

RefPtr<CSSValue> consumeClipPath(CSSParserTokenRange& range, const CSSParserContext& context)
{
    // <'clip-path'> = none | <clip-source> | [ <basic-shape> || <geometry-box> ]
    // <clip-source> = <url>
    // https://drafts.fxtf.org/css-masking/#propdef-clip-path

    if (range.peek().id() == CSSValueNone)
        return consumeIdent(range);

    if (auto url = consumeURL(range))
        return url;

    RefPtr<CSSValue> shape;
    RefPtr<CSSValue> box;

    auto consumeShape = [&]() -> bool {
        if (shape)
            return false;
        shape = consumeBasicShape(range, context, { });
        return !!shape;
    };
    auto consumeBox = [&]() -> bool {
        if (box)
            return false;
        box = CSSPropertyParsing::consumeGeometryBox(range);
        return !!box;
    };

    while (!range.atEnd()) {
        if (consumeShape() || consumeBox())
            continue;
        break;
    }

    bool hasShape = !!shape;

    CSSValueListBuilder list;
    if (shape)
        list.append(shape.releaseNonNull());
    // Default value is border-box.
    if (box && (box->valueID() != CSSValueBorderBox || !hasShape))
        list.append(box.releaseNonNull());

    if (list.isEmpty())
        return nullptr;

    return CSSValueList::createSpaceSeparated(WTFMove(list));
}

} // namespace CSSPropertyParserHelpers
} // namespace WebCore
