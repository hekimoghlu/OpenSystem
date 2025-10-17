/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 14, 2025.
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

#include <wtf/StdLibExtras.h>
#include <wtf/text/StringCommon.h>
#include <wtf/text/StringParsingBuffer.h>

namespace WTF {

template<typename CharacterType> inline bool isNotASCIISpace(CharacterType c)
{
    return !isUnicodeCompatibleASCIIWhitespace(c);
}

template<typename T> void skip(std::span<T>& data, size_t amountToSkip)
{
    data = data.subspan(amountToSkip);
}

template<typename T> void dropLast(std::span<T>& data, size_t amountToDrop = 1)
{
    data = data.first(data.size() - amountToDrop);
}

template<typename T> T& consumeLast(std::span<T>& data)
{
    auto* last = &data.back();
    data = data.first(data.size() - 1);
    return *last;
}

template<typename T> void clampedMoveCursorWithinSpan(std::span<T>& cursor, std::span<T> container, int delta)
{
    ASSERT(cursor.data() >= container.data());
    ASSERT(std::to_address(cursor.end()) == std::to_address(container.end()));
    auto clampedNewIndex = std::clamp<int>(cursor.data() - container.data() + delta, 0, container.size());
    cursor = container.subspan(clampedNewIndex);
}

template<typename CharacterType, typename DelimiterType> bool skipExactly(const CharacterType*& position, const CharacterType* end, DelimiterType delimiter)
{
    if (position < end && *position == delimiter) {
        ++position;
        return true;
    }
    return false;
}

template<typename CharacterType, typename DelimiterType> bool skipExactly(std::span<CharacterType>& data, DelimiterType delimiter)
{
    if (!data.empty() && data.front() == delimiter) {
        skip(data, 1);
        return true;
    }
    return false;
}

template<typename CharacterType, typename DelimiterType> bool skipExactly(StringParsingBuffer<CharacterType>& buffer, DelimiterType delimiter)
{
    if (buffer.hasCharactersRemaining() && *buffer == delimiter) {
        ++buffer;
        return true;
    }
    return false;
}

template<bool characterPredicate(LChar)> bool skipExactly(StringParsingBuffer<LChar>& buffer)
{
    if (buffer.hasCharactersRemaining() && characterPredicate(*buffer)) {
        ++buffer;
        return true;
    }
    return false;
}

template<bool characterPredicate(UChar)> bool skipExactly(StringParsingBuffer<UChar>& buffer)
{
    if (buffer.hasCharactersRemaining() && characterPredicate(*buffer)) {
        ++buffer;
        return true;
    }
    return false;
}

template<bool characterPredicate(LChar), typename CharacterType> bool skipExactly(std::span<CharacterType>& buffer) requires(std::is_same_v<std::remove_const_t<CharacterType>, LChar>)
{
    if (!buffer.empty() && characterPredicate(buffer[0])) {
        skip(buffer, 1);
        return true;
    }
    return false;
}

template<bool characterPredicate(UChar), typename CharacterType> bool skipExactly(std::span<CharacterType>& buffer) requires(std::is_same_v<std::remove_const_t<CharacterType>, UChar>)
{
    if (!buffer.empty() && characterPredicate(buffer[0])) {
        skip(buffer, 1);
        return true;
    }
    return false;
}

template<typename CharacterType, typename DelimiterType> void skipUntil(StringParsingBuffer<CharacterType>& buffer, DelimiterType delimiter)
{
    while (buffer.hasCharactersRemaining() && *buffer != delimiter)
        ++buffer;
}

template<typename CharacterType, typename DelimiterType> void skipUntil(std::span<CharacterType>& buffer, DelimiterType delimiter)
{
    size_t index = 0;
    while (index < buffer.size() && buffer[index] != delimiter)
        ++index;
    skip(buffer, index);
}

template<bool characterPredicate(LChar), typename CharacterType> void skipUntil(std::span<CharacterType>& data) requires(std::is_same_v<std::remove_const_t<CharacterType>, LChar>)
{
    size_t index = 0;
    while (index < data.size() && !characterPredicate(data[index]))
        ++index;
    skip(data, index);
}

template<bool characterPredicate(UChar), typename CharacterType> void skipUntil(std::span<CharacterType>& data) requires(std::is_same_v<std::remove_const_t<CharacterType>, UChar>)
{
    size_t index = 0;
    while (index < data.size() && !characterPredicate(data[index]))
        ++index;
    skip(data, index);
}

template<bool characterPredicate(LChar)> void skipUntil(StringParsingBuffer<LChar>& buffer)
{
    while (buffer.hasCharactersRemaining() && !characterPredicate(*buffer))
        ++buffer;
}

template<bool characterPredicate(UChar)> void skipUntil(StringParsingBuffer<UChar>& buffer)
{
    while (buffer.hasCharactersRemaining() && !characterPredicate(*buffer))
        ++buffer;
}

template<typename CharacterType, typename DelimiterType> void skipWhile(StringParsingBuffer<CharacterType>& buffer, DelimiterType delimiter)
{
    while (buffer.hasCharactersRemaining() && *buffer == delimiter)
        ++buffer;
}

template<typename CharacterType, typename DelimiterType> void skipWhile(std::span<CharacterType>& buffer, DelimiterType delimiter)
{
    size_t index = 0;
    while (index < buffer.size() && buffer[index] == delimiter)
        ++index;
    skip(buffer, index);
}

template<bool characterPredicate(LChar), typename CharacterType> void skipWhile(std::span<CharacterType>& data) requires(std::is_same_v<std::remove_const_t<CharacterType>, LChar>)
{
    size_t index = 0;
    while (index < data.size() && characterPredicate(data[index]))
        ++index;
    skip(data, index);
}

template<bool characterPredicate(UChar), typename CharacterType> void skipWhile(std::span<CharacterType>& data) requires(std::is_same_v<std::remove_const_t<CharacterType>, UChar>)
{
    size_t index = 0;
    while (index < data.size() && characterPredicate(data[index]))
        ++index;
    skip(data, index);
}

template<bool characterPredicate(LChar)> void skipWhile(StringParsingBuffer<LChar>& buffer)
{
    while (buffer.hasCharactersRemaining() && characterPredicate(*buffer))
        ++buffer;
}

template<bool characterPredicate(UChar)> void skipWhile(StringParsingBuffer<UChar>& buffer)
{
    while (buffer.hasCharactersRemaining() && characterPredicate(*buffer))
        ++buffer;
}

template<typename CharacterType> bool skipExactlyIgnoringASCIICase(StringParsingBuffer<CharacterType>& buffer, ASCIILiteral literal)
{
    auto literalLength = literal.length();

    if (buffer.lengthRemaining() < literalLength)
        return false;
    if (!equalLettersIgnoringASCIICaseWithLength(buffer.span(), literal.span8(), literalLength))
        return false;
    buffer += literalLength;
    return true;
}

template<typename CharacterType, std::size_t Extent> bool skipLettersExactlyIgnoringASCIICase(StringParsingBuffer<CharacterType>& buffer, std::span<const CharacterType, Extent> letters)
{
    if (buffer.lengthRemaining() < letters.size())
        return false;
    for (unsigned i = 0; i < letters.size(); ++i) {
        ASSERT(isASCIIAlpha(letters[i]));
        if (!isASCIIAlphaCaselessEqual(buffer[i], static_cast<char>(letters[i])))
            return false;
    }
    buffer += letters.size();
    return true;
}

template<typename CharacterType, std::size_t Extent> bool skipLettersExactlyIgnoringASCIICase(std::span<const CharacterType>& buffer, std::span<const CharacterType, Extent> letters)
{
    if (buffer.size() < letters.size())
        return false;
    if (!equalLettersIgnoringASCIICaseWithLength(buffer, letters, letters.size()))
        return false;
    skip(buffer, letters.size());
    return true;
}

template<typename CharacterType, std::size_t Extent> constexpr bool skipCharactersExactly(StringParsingBuffer<CharacterType>& buffer, std::span<const CharacterType, Extent> string)
{
    if (!spanHasPrefix(buffer.span(), string))
        return false;
    buffer += string.size();
    return true;
}

template<typename CharacterType, std::size_t Extent> constexpr bool skipCharactersExactly(std::span<const CharacterType>& buffer, std::span<const CharacterType, Extent> string)
{
    if (!spanHasPrefix(buffer, string))
        return false;
    skip(buffer, string.size());
    return true;
}

template<typename T> std::span<T> consumeSpan(std::span<T>& data, size_t amountToConsume)
{
    auto consumed = data.first(amountToConsume);
    skip(data, amountToConsume);
    return consumed;
}

template<typename T> T& consume(std::span<T>& data)
{
    T& value = data[0];
    skip(data, 1);
    return value;
}

template<typename DestinationType, typename SourceType>
match_constness_t<SourceType, DestinationType>& consumeAndCastTo(std::span<SourceType>& data) requires(sizeof(SourceType) == 1)
{
    return spanReinterpretCast<match_constness_t<SourceType, DestinationType>>(consumeSpan(data, sizeof(DestinationType)))[0];
}

// Adapt a UChar-predicate to an LChar-predicate.
template<bool characterPredicate(UChar)>
static inline bool LCharPredicateAdapter(LChar c) { return characterPredicate(c); }

} // namespace WTF

using WTF::LCharPredicateAdapter;
using WTF::clampedMoveCursorWithinSpan;
using WTF::consume;
using WTF::consumeAndCastTo;
using WTF::consumeLast;
using WTF::consumeSpan;
using WTF::dropLast;
using WTF::isNotASCIISpace;
using WTF::skip;
using WTF::skipCharactersExactly;
using WTF::skipExactly;
using WTF::skipExactlyIgnoringASCIICase;
using WTF::skipLettersExactlyIgnoringASCIICase;
using WTF::skipUntil;
using WTF::skipWhile;
