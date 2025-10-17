/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 24, 2022.
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

#include <wtf/HashFunctions.h>
#include <wtf/HashTraits.h>
#include <wtf/text/OrdinalNumber.h>

namespace WTF {

// TextPosition structure specifies coordinates within an text resource. It is used mostly
// for saving script source position.
class TextPosition {
    WTF_MAKE_FAST_ALLOCATED;
public:
    TextPosition(OrdinalNumber line, OrdinalNumber column)
        : m_line(line)
        , m_column(column)
    {
    }

    TextPosition() { }
    friend bool operator==(const TextPosition&, const TextPosition&) = default;
    friend std::strong_ordering operator<=>(const TextPosition& a, const TextPosition& b)
    {
        auto lineComparison = a.m_line <=> b.m_line;
        return lineComparison != std::strong_ordering::equal ? lineComparison : a.m_column <=> b.m_column;
    }

    // A value with line value less than a minimum; used as an impossible position.
    static TextPosition belowRangePosition() { return TextPosition(OrdinalNumber::beforeFirst(), OrdinalNumber::beforeFirst()); }

    OrdinalNumber m_line;
    OrdinalNumber m_column;
};

template<typename T> struct DefaultHash;
template<> struct DefaultHash<TextPosition> {
    static unsigned hash(const TextPosition& key) { return pairIntHash(static_cast<unsigned>(key.m_line.zeroBasedInt()), static_cast<unsigned>(key.m_column.zeroBasedInt())); }
    static bool equal(const TextPosition& a, const TextPosition& b) { return a == b; }
    static constexpr bool safeToCompareToEmptyOrDeleted = true;
};

template<typename T> struct HashTraits;
template<> struct HashTraits<TextPosition> : GenericHashTraits<TextPosition> {
    static void constructDeletedValue(TextPosition& slot)
    {
        slot = TextPosition::belowRangePosition();
    }
    static bool isDeletedValue(const TextPosition& value)
    {
        return value == TextPosition::belowRangePosition();
    }
};

}

using WTF::TextPosition;
