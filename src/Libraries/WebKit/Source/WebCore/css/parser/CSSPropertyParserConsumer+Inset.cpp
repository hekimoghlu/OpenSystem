/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 11, 2024.
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
#include "CSSPropertyParserConsumer+Inset.h"

#include "CSSParserContext.h"
#include "CSSParserTokenRange.h"
#include "CSSPrimitiveValue.h"
#include "CSSPropertyParserConsumer+Ident.h"
#include "CSSPropertyParserConsumer+LengthPercentage.h"
#include "CSSValueKeywords.h"

namespace WebCore {
namespace CSSPropertyParserHelpers {

RefPtr<CSSValue> consumeInsetPhysical(CSSParserTokenRange& range, const CSSParserContext& context, CSSPropertyID currentShorthand)
{
    // <inset-physical> = auto | <length-percentage anchor-allowed anchor-size-allowed>
    // https://drafts.csswg.org/css-position-3/#insets

    if (range.peek().id() == CSSValueAuto)
        return consumeIdent(range);

    auto unitless = currentShorthand != CSSPropertyInset ? UnitlessQuirk::Allow : UnitlessQuirk::Forbid;
    return consumeLengthPercentage(range, context, ValueRange::All, unitless, UnitlessZeroQuirk::Allow, AnchorPolicy::Allow, AnchorSizePolicy::Allow);
}

} // namespace CSSPropertyParserHelpers
} // namespace WebCore
