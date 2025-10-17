/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 10, 2024.
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

#include <wtf/text/StringView.h>

namespace WTF {

class NullTextBreakIterator {
    WTF_MAKE_FAST_ALLOCATED;
public:
    NullTextBreakIterator() = default;
    NullTextBreakIterator(const NullTextBreakIterator&) = delete;
    NullTextBreakIterator(NullTextBreakIterator&&) = default;
    NullTextBreakIterator& operator=(const NullTextBreakIterator&) = delete;
    NullTextBreakIterator& operator=(NullTextBreakIterator&&) = default;

    std::optional<unsigned> preceding(unsigned) const
    {
        ASSERT_NOT_REACHED();
        return { };
    }

    std::optional<unsigned> following(unsigned) const
    {
        ASSERT_NOT_REACHED();
        return { };
    }

    bool isBoundary(unsigned) const
    {
        ASSERT_NOT_REACHED();
        return false;
    }

    void setText(StringView, std::span<const UChar>)
    {
        ASSERT_NOT_REACHED();
    }
};

}

