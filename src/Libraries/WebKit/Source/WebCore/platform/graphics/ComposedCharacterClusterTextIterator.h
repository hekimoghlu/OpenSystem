/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 15, 2024.
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

#include <unicode/utf16.h>
#include <wtf/text/TextBreakIterator.h>

namespace WebCore {

class ComposedCharacterClusterTextIterator {
public:
    // The passed in UChar pointer starts at 'currentIndex'. The iterator operates on the range [currentIndex, lastIndex].
    ComposedCharacterClusterTextIterator(std::span<const UChar> characters, unsigned currentIndex, unsigned lastIndex)
        : m_iterator(characters, { }, TextBreakIterator::CaretMode { }, nullAtom())
        , m_characters(characters)
        , m_originalIndex(currentIndex)
        , m_currentIndex(currentIndex)
        , m_lastIndex(lastIndex)
    {
    }

    bool consume(char32_t& character, unsigned& clusterLength)
    {
        if (m_currentIndex >= m_lastIndex)
            return false;

        auto relativeIndex = m_currentIndex - m_originalIndex;
        if (auto result = m_iterator.following(relativeIndex)) {
            clusterLength = result.value() - relativeIndex;
            U16_NEXT(m_characters, relativeIndex, result.value(), character);
            return true;
        }
        
        return false;
    }

    void advance(unsigned advanceLength)
    {
        m_currentIndex += advanceLength;
    }

    void reset(unsigned index)
    {
        ASSERT(index >= m_originalIndex);
        if (index >= m_lastIndex)
            return;
        m_currentIndex = index;
    }

    std::span<const UChar> remainingCharacters() const
    {
        auto relativeIndex = m_currentIndex - m_originalIndex;
        return m_characters.subspan(relativeIndex);
    }

    unsigned currentIndex() const { return m_currentIndex; }
    std::span<const UChar> characters() const { return m_characters; }

private:
    CachedTextBreakIterator m_iterator;
    std::span<const UChar> m_characters;
    const unsigned m_originalIndex { 0 };
    unsigned m_currentIndex { 0 };
    const unsigned m_lastIndex { 0 };
};

} // namespace WebCore
