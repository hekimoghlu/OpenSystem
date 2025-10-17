/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 25, 2022.
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
#include "TextBoundaries.h"

#include <unicode/ubrk.h>
#include <wtf/text/StringImpl.h>
#include <wtf/text/TextBreakIterator.h>

namespace WebCore {

unsigned endOfFirstWordBoundaryContext(StringView text)
{
    unsigned length = text.length();
    for (unsigned i = 0; i < length; ) {
        unsigned first = i;
        char32_t ch;
        U16_NEXT(text, i, length, ch);
        if (!requiresContextForWordBoundary(ch))
            return first;
    }
    return length;
}

unsigned startOfLastWordBoundaryContext(StringView text)
{
    unsigned length = text.length();
    for (unsigned i = length; i > 0; ) {
        unsigned last = i;
        char32_t ch;
        U16_PREV(text, 0, i, ch);
        if (!requiresContextForWordBoundary(ch))
            return last;
    }
    return 0;
}

#if !USE(APPKIT)

int findNextWordFromIndex(StringView text, int position, bool forward)
{
    UBreakIterator* it = wordBreakIterator(text);

    if (forward) {
        position = ubrk_following(it, position);
        while (position != UBRK_DONE) {
            // We stop searching when the character preceeding the break is alphanumeric.
            if (static_cast<unsigned>(position) < text.length() && u_isalnum(text[position - 1]))
                return position;

            position = ubrk_following(it, position);
        }

        return text.length();
    } else {
        position = ubrk_preceding(it, position);
        while (position != UBRK_DONE) {
            // We stop searching when the character following the break is alphanumeric.
            if (position && u_isalnum(text[position]))
                return position;

            position = ubrk_preceding(it, position);
        }

        return 0;
    }
}

#endif // !USE(APPKIT)

#if !PLATFORM(COCOA)

void findWordBoundary(StringView text, int position, int* start, int* end)
{
    UBreakIterator* it = wordBreakIterator(text);
    *end = ubrk_following(it, position);
    if (*end < 0)
        *end = ubrk_last(it);
    *start = ubrk_previous(it);
}

void findEndWordBoundary(StringView text, int position, int* end)
{
    UBreakIterator* it = wordBreakIterator(text);
    *end = ubrk_following(it, position);
    if (*end < 0)
        *end = ubrk_last(it);
}

#endif // !PLATFORM(COCOA)

} // namespace WebCore
