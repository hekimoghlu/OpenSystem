/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 10, 2022.
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
#include "LegacyInlineIterator.h"

#include "RenderStyleInlines.h"

namespace WebCore {

UCharDirection LegacyInlineIterator::surrogateTextDirection(UChar currentCodeUnit) const
{
    RenderText& text = downcast<RenderText>(*m_renderer);
    UChar lead;
    UChar trail;
    if (U16_IS_LEAD(currentCodeUnit)) {
        lead = currentCodeUnit;
        trail = text.characterAt(m_pos + 1);
        if (!U16_IS_TRAIL(trail))
            return U_OTHER_NEUTRAL;
    } else {
        ASSERT(U16_IS_TRAIL(currentCodeUnit));
        lead = text.characterAt(m_pos - 1);
        if (!U16_IS_LEAD(lead))
            return U_OTHER_NEUTRAL;
        trail = currentCodeUnit;
    }
    return u_charDirection(U16_GET_SUPPLEMENTARY(lead, trail));
}

}
