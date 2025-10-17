/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 7, 2023.
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
#include "CSSPropertyParserConsumer+UI.h"

#include "CSSCursorImageValue.h"
#include "CSSParserContext.h"
#include "CSSParserTokenRange.h"
#include "CSSPrimitiveValue.h"
#include "CSSPropertyParserConsumer+Ident.h"
#include "CSSPropertyParserConsumer+Image.h"
#include "CSSPropertyParserConsumer+Number.h"
#include "CSSPropertyParserConsumer+Primitives.h"
#include "CSSValueKeywords.h"
#include "CSSValueList.h"
#include "CSSValuePair.h"

namespace WebCore {
namespace CSSPropertyParserHelpers {

RefPtr<CSSValue> consumeCursor(CSSParserTokenRange& range, const CSSParserContext& context)
{
    // <cursor> = [ [ <url> | <url-set> ] [<x> <y>]? ]#? [ auto | default | none | context-menu | help | pointer | progress | wait | cell | crosshair | text | vertical-text | alias | copy | move | no-drop | not-allowed | grab | grabbing | e-resize | n-resize | ne-resize | nw-resize | s-resize | se-resize | sw-resize | w-resize | ew-resize | ns-resize | nesw-resize | nwse-resize | col-resize | row-resize | all-scroll | zoom-in | zoom-out ]
    // https://drafts.csswg.org/css-ui/#propdef-cursor

    CSSValueListBuilder list;
    while (auto image = consumeImage(range, context, { AllowedImageType::URLFunction, AllowedImageType::ImageSet })) {
        RefPtr<CSSValuePair> hotSpot;
        if (auto x = consumeNumber(range, context)) {
            auto y = consumeNumber(range, context);
            if (!y)
                return nullptr;
            hotSpot = CSSValuePair::createNoncoalescing(x.releaseNonNull(), y.releaseNonNull());
        }
        list.append(CSSCursorImageValue::create(image.releaseNonNull(), WTFMove(hotSpot), context.isContentOpaque ? LoadedFromOpaqueSource::Yes : LoadedFromOpaqueSource::No));
        if (!consumeCommaIncludingWhitespace(range))
            return nullptr;
    }

    CSSValueID id = range.peek().id();
    RefPtr<CSSValue> cursorType;
    if (id == CSSValueHand) {
        if (context.mode != HTMLQuirksMode) // Non-standard behavior
            return nullptr;
        cursorType = CSSPrimitiveValue::create(CSSValuePointer);
        range.consumeIncludingWhitespace();
    } else if ((id >= CSSValueAuto && id <= CSSValueWebkitZoomOut) || id == CSSValueCopy || id == CSSValueNone)
        cursorType = consumeIdent(range);
    else
        return nullptr;

    if (list.isEmpty())
        return cursorType;
    list.append(cursorType.releaseNonNull());
    return CSSValueList::createCommaSeparated(WTFMove(list));
}

} // namespace CSSPropertyParserHelpers
} // namespace WebCore
