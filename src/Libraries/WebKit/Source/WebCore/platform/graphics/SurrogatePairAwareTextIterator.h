/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 2, 2023.
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
#include <wtf/text/WTFString.h>

namespace WebCore {

class SurrogatePairAwareTextIterator {
public:
    // The passed in UChar pointer starts at 'currentIndex'. The iterator operates on the range [currentIndex, lastIndex].
    // 'endIndex' denotes the maximum length of the UChar array, which might exceed 'lastIndex'.
    SurrogatePairAwareTextIterator(std::span<const UChar> characters, unsigned currentIndex, unsigned lastIndex)
        : m_characters(characters)
        , m_currentIndex(currentIndex)
        , m_originalIndex(currentIndex)
        , m_lastIndex(lastIndex)
        , m_endIndex(characters.size() + currentIndex)
    {
    }

    bool consume(char32_t& character, unsigned& clusterLength)
    {
        if (m_currentIndex >= m_lastIndex)
            return false;

        auto relativeIndex = m_currentIndex - m_originalIndex;
        clusterLength = 0;
        auto spanAtRelativeIndex = m_characters.subspan(relativeIndex);
        U16_NEXT(spanAtRelativeIndex, clusterLength, m_endIndex - m_currentIndex, character);
        return true;
    }

    void advance(unsigned advanceLength)
    {
        m_currentIndex += advanceLength;
    }

    void reset(unsigned index)
    {
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
    std::span<const UChar> m_characters;
    unsigned m_currentIndex { 0 };
    const unsigned m_originalIndex { 0 };
    const unsigned m_lastIndex { 0 };
    const unsigned m_endIndex { 0 };
};

} // namespace WebCore
