/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 18, 2023.
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

#include <unicode/uchar.h>
#include <wtf/Forward.h>

namespace WebCore {

inline bool requiresContextForWordBoundary(char32_t character)
{
    // FIXME: This function won't be needed when https://bugs.webkit.org/show_bug.cgi?id=120656 is fixed.

    // We can get rid of this constant if we require a newer version of the ICU headers, which has U_LB_CONDITIONAL_JAPANESE_STARTER.
    constexpr int lineBreakConditionalJapaneseStarter = 37;

    int lineBreak = u_getIntPropertyValue(character, UCHAR_LINE_BREAK);
    return lineBreak == U_LB_COMPLEX_CONTEXT || lineBreak == lineBreakConditionalJapaneseStarter || lineBreak == U_LB_IDEOGRAPHIC;
}

unsigned endOfFirstWordBoundaryContext(StringView);
unsigned startOfLastWordBoundaryContext(StringView);

WEBCORE_EXPORT void findWordBoundary(StringView, int position, int* start, int* end);
void findEndWordBoundary(StringView, int position, int* end);
int findNextWordFromIndex(StringView, int position, bool forward);

}
