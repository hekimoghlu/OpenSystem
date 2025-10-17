/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 15, 2024.
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

template<typename T>
class StringParsingBuffer final {
    WTF_MAKE_FAST_ALLOCATED;
public:
    using CharacterType = T;

    constexpr StringParsingBuffer() = default;

    constexpr StringParsingBuffer(std::span<const CharacterType> characters LIFETIME_BOUND)
        : m_data { characters }
    {
        ASSERT(m_data.data() || m_data.empty());
    }

    constexpr auto position() const LIFETIME_BOUND { return m_data.data(); }
    constexpr auto end() const LIFETIME_BOUND { return std::to_address(m_data.end()); }

    constexpr bool hasCharactersRemaining() const { return !m_data.empty(); }
    constexpr bool atEnd() const { return m_data.empty(); }

    constexpr size_t lengthRemaining() const { return m_data.size(); }

    constexpr void setPosition(std::span<const CharacterType> position)
    {
        ASSERT(position.data() <= std::to_address(m_data.end()));
        ASSERT(std::to_address(position.end()) <= std::to_address(m_data.end()));
        m_data = position;
    }

    StringView stringViewOfCharactersRemaining() const LIFETIME_BOUND { return span(); }

    CharacterType consume()
    {
        ASSERT(hasCharactersRemaining());
        auto character = m_data.front();
        m_data = m_data.subspan(1);
        return character;
    }

    std::span<const CharacterType> span() const LIFETIME_BOUND { return m_data; }

    std::span<const CharacterType> consume(size_t count) LIFETIME_BOUND
    {
        ASSERT(count <= lengthRemaining());
        auto result = m_data;
        m_data = m_data.subspan(count);
        return result;
    }

    CharacterType operator[](size_t i) const
    {
        ASSERT(i < lengthRemaining());
        return m_data[i];
    }

    constexpr CharacterType operator*() const
    {
        ASSERT(hasCharactersRemaining());
        return m_data.front();
    }

    constexpr void advance()
    {
        ASSERT(hasCharactersRemaining());
        m_data = m_data.subspan(1);
    }

    constexpr void advanceBy(size_t places)
    {
        ASSERT(places <= lengthRemaining());
        m_data = m_data.subspan(places);
    }

    constexpr StringParsingBuffer& operator++()
    {
        advance();
        return *this;
    }

    constexpr StringParsingBuffer operator++(int)
    {
        auto result = *this;
        ++*this;
        return result;
    }

    constexpr StringParsingBuffer& operator+=(int places)
    {
        advanceBy(places);
        return *this;
    }

private:
    std::span<const CharacterType> m_data;
};

template<typename StringType, typename Function> decltype(auto) readCharactersForParsing(StringType&& string, Function&& functor)
{
    if (string.is8Bit())
        return functor(StringParsingBuffer { string.span8() });
    return functor(StringParsingBuffer { string.span16() });
}

} // namespace WTF

using WTF::StringParsingBuffer;
using WTF::readCharactersForParsing;
