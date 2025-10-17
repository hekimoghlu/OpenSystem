/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 22, 2023.
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

#include <unicode/utypes.h>
#include <wtf/Assertions.h>
#include <wtf/text/LChar.h>
#include <wtf/text/ParsingUtilities.h>

namespace WTF {

template<typename CharacterType>
class CodePointIterator {
    WTF_MAKE_FAST_ALLOCATED;
public:
    ALWAYS_INLINE CodePointIterator() = default;
    ALWAYS_INLINE CodePointIterator(std::span<const CharacterType> data)
        : m_data(data)
    {
    }
    
WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN
    ALWAYS_INLINE CodePointIterator(const CodePointIterator& begin, const CodePointIterator& end)
        : CodePointIterator({ begin.m_data.data(), end.m_data.data() })
    {
        ASSERT(end.m_data.data() >= begin.m_data.data());
    }
WTF_ALLOW_UNSAFE_BUFFER_USAGE_END
    
    ALWAYS_INLINE char32_t operator*() const;
    ALWAYS_INLINE CodePointIterator& operator++();

    ALWAYS_INLINE friend bool operator==(const CodePointIterator& a, const CodePointIterator& b)
    {
        return a.m_data.data() == b.m_data.data() && a.m_data.size() == b.m_data.size();
    }

    ALWAYS_INLINE bool atEnd() const
    {
        return m_data.empty();
    }
    
    ALWAYS_INLINE size_t codeUnitsSince(const CharacterType* reference) const
    {
        ASSERT(m_data.data() >= reference);
        return m_data.data() - reference;
    }

    ALWAYS_INLINE size_t codeUnitsSince(const CodePointIterator& other) const
    {
        return codeUnitsSince(other.m_data.data());
    }
    
private:
    std::span<const CharacterType> m_data;
};

template<>
ALWAYS_INLINE char32_t CodePointIterator<LChar>::operator*() const
{
    ASSERT(!atEnd());
    return m_data.front();
}

template<>
ALWAYS_INLINE auto CodePointIterator<LChar>::operator++() -> CodePointIterator&
{
    skip(m_data, 1);
    return *this;
}

template<>
ALWAYS_INLINE char32_t CodePointIterator<UChar>::operator*() const
{
    ASSERT(!atEnd());
    char32_t c;
    U16_GET(m_data, 0, 0, m_data.size(), c);
    return c;
}

template<>
ALWAYS_INLINE auto CodePointIterator<UChar>::operator++() -> CodePointIterator&
{
    unsigned i = 0;
    size_t length = m_data.size();
    U16_FWD_1(m_data, i, length);
    skip(m_data, i);
    return *this;
}

template<typename CharacterType> CodePointIterator(std::span<const CharacterType>) -> CodePointIterator<CharacterType>;

} // namespace WTF
