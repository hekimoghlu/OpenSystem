/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 16, 2025.
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
#include "StylePropertyShorthand.h"
#include "StylePropertyShorthandFunctions.h"

namespace WebCore {

StylePropertyShorthand transitionShorthandForParsing()
{
    // Similar to animations, we have property after timing-function and delay after
    // duration.
    static constexpr std::array transitionProperties = {
        CSSPropertyTransitionDuration, CSSPropertyTransitionTimingFunction,
        CSSPropertyTransitionDelay, CSSPropertyTransitionBehavior, CSSPropertyTransitionProperty };
    return StylePropertyShorthand(CSSPropertyTransition, std::span { transitionProperties });
}

unsigned indexOfShorthandForLonghand(CSSPropertyID shorthandID, const StylePropertyShorthandVector& shorthands)
{
    for (unsigned i = 0, size = shorthands.size(); i < size; ++i) {
        if (shorthands[i].id() == shorthandID)
            return i;
    }
    ASSERT_NOT_REACHED();
    return 0;
}

} // namespace WebCore
