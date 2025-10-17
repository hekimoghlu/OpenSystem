/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 31, 2021.
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

#include <limits>
#include <wtf/text/StringCommon.h>
#include <wtf/text/StringView.h>

namespace WTF {

template<typename OffsetType>
class BoyerMooreHorspoolTable {
    WTF_MAKE_FAST_ALLOCATED(BoyerMooreHorspoolTable);
public:
    static constexpr unsigned size = 256;
    static constexpr unsigned maxPatternLength = std::numeric_limits<OffsetType>::max();

    explicit BoyerMooreHorspoolTable(StringView pattern)
    {
        if (pattern.is8Bit())
            initializeTable(pattern.span8());
        else
            initializeTable(pattern.span16());
    }

    explicit constexpr BoyerMooreHorspoolTable(ASCIILiteral pattern)
    {
        initializeTable(pattern.span8());
    }

    ALWAYS_INLINE size_t find(StringView string, StringView matchString) const
    {
        unsigned matchLength = matchString.length();
        if (matchLength > string.length())
            return notFound;

        if (UNLIKELY(!matchLength))
            return 0;

        if (string.is8Bit()) {
            if (matchString.is8Bit())
                return findInner(string.span8(), matchString.span8());
            return findInner(string.span8(), matchString.span16());
        }

        if (matchString.is8Bit())
            return findInner(string.span16(), matchString.span8());
        return findInner(string.span16(), matchString.span16());
    }

private:
    template<typename CharacterType>
    constexpr void initializeTable(std::span<CharacterType> pattern)
    {
        size_t length = pattern.size();
        ASSERT_UNDER_CONSTEXPR_CONTEXT(length <= maxPatternLength);
        if (length) {
            for (auto& element : m_table)
                element = length;
            for (unsigned i = 0; i < (pattern.size() - 1); ++i) {
                unsigned index = pattern[i] & 0xff;
                m_table[index] = length - 1 - i;
            }
        }
    }

    template <typename SearchCharacterType, typename MatchCharacterType>
    ALWAYS_INLINE size_t findInner(std::span<const SearchCharacterType> characters, std::span<const MatchCharacterType> matchCharacters) const
    {
        size_t cursor = 0;
        size_t last = characters.size() - matchCharacters.size();
        while (cursor <= last) {
            if (equal(characters.subspan(cursor).data(), matchCharacters))
                return cursor;
            cursor += m_table[static_cast<uint8_t>(characters[cursor + matchCharacters.size() - 1])];
        }
        return notFound;
    }

    std::array<OffsetType, size> m_table;
};

}

using WTF::BoyerMooreHorspoolTable;
