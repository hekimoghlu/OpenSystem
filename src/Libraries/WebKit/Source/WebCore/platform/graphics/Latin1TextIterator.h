/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 16, 2022.
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

#include <wtf/text/WTFString.h>

namespace WebCore {

class Latin1TextIterator {
public:
    // The passed in LChar pointer starts at 'currentIndex'. The iterator operates on the range [currentIndex, lastIndex].
    // 'endCharacter' denotes the maximum length of the UChar array, which might exceed 'lastIndex'.
    Latin1TextIterator(std::span<const LChar> characters, unsigned currentIndex, unsigned lastIndex)
        : m_characters(characters)
        , m_currentIndex(currentIndex)
        , m_originalIndex(currentIndex)
        , m_lastIndex(lastIndex)
    {
    }

    bool consume(char32_t& character, unsigned& clusterLength)
    {
        if (m_currentIndex >= m_lastIndex)
            return false;

        auto relativeIndex = m_currentIndex - m_originalIndex;
        character = m_characters[relativeIndex];
        clusterLength = 1;
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

    std::span<const LChar> remainingCharacters() const
    {
        auto relativeIndex = m_currentIndex - m_originalIndex;
        return m_characters.subspan(relativeIndex);
    }

    unsigned currentIndex() const { return m_currentIndex; }
    std::span<const LChar> characters() const { return m_characters; }

private:
    std::span<const LChar> m_characters;
    unsigned m_currentIndex;
    const unsigned m_originalIndex;
    const unsigned m_lastIndex;
};

} // namespace WebCore
