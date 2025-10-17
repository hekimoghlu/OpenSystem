/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 4, 2021.
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
#include <wtf/unicode/UTF8Conversion.h>

#include <unicode/uchar.h>
#include <wtf/ASCIICType.h>
#include <wtf/text/StringHasherInlines.h>
#include <wtf/unicode/CharacterNames.h>

namespace WTF::Unicode {

static constexpr char32_t sentinelCodePoint = U_SENTINEL;

enum class Replacement : bool { None, ReplaceInvalidSequences };

template<Replacement = Replacement::None, typename CharacterType> static char32_t next(std::span<const CharacterType>, size_t& offset);
template<Replacement = Replacement::None, typename CharacterType> static bool append(std::span<CharacterType>, size_t& offset, char32_t character);

template<> char32_t next<Replacement::None, LChar>(std::span<const LChar> characters, size_t& offset)
{
    return characters[offset++];
}

template<> char32_t next<Replacement::None, char8_t>(std::span<const char8_t> characters, size_t& offset)
{
    char32_t character;
WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN
    U8_NEXT(characters, offset, characters.size(), character);
WTF_ALLOW_UNSAFE_BUFFER_USAGE_END
    return U_IS_SURROGATE(character) ? sentinelCodePoint : character;
}

template<> char32_t next<Replacement::ReplaceInvalidSequences, char8_t>(std::span<const char8_t> characters, size_t& offset)
{
    char32_t character;
WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN
    U8_NEXT_OR_FFFD(characters, offset, characters.size(), character);
WTF_ALLOW_UNSAFE_BUFFER_USAGE_END
    return character;
}

template<> char32_t next<Replacement::None, char16_t>(std::span<const char16_t> characters, size_t& offset)
{
    char32_t character;
    U16_NEXT(characters, offset, characters.size(), character);
    return U_IS_SURROGATE(character) ? sentinelCodePoint : character;
}

template<> char32_t next<Replacement::ReplaceInvalidSequences, char16_t>(std::span<const char16_t> characters, size_t& offset)
{
    char32_t character;
    U16_NEXT_OR_FFFD(characters, offset, characters.size(), character);
    return character;
}

template<> bool append<Replacement::None, char8_t>(std::span<char8_t> characters, size_t& offset, char32_t character)
{
    UBool sawError = false;
    U8_APPEND(characters, offset, characters.size(), character, sawError);
    return sawError;
}

template<> bool append<Replacement::ReplaceInvalidSequences, char8_t>(std::span<char8_t> characters, size_t& offset, char32_t character)
{
    return append(characters, offset, character)
        && append(characters, offset, replacementCharacter);
}

template<> bool append<Replacement::None, char16_t>(std::span<char16_t> characters, size_t& offset, char32_t character)
{
    UBool sawError = false;
    U16_APPEND(characters, offset, characters.size(), character, sawError);
    return sawError;
}

template<> bool append<Replacement::ReplaceInvalidSequences, char16_t>(std::span<char16_t> characters, size_t& offset, char32_t character)
{
    return append(characters, offset, character)
        && append(characters, offset, replacementCharacter);
}

template<Replacement replacement = Replacement::None, typename SourceCharacterType, typename BufferCharacterType> static ConversionResult<BufferCharacterType> convertInternal(std::span<const SourceCharacterType> source, std::span<BufferCharacterType> buffer)
{
    auto resultCode = ConversionResultCode::Success;
    size_t bufferOffset = 0;
    char32_t orAllData = 0;
    for (size_t sourceOffset = 0; sourceOffset < source.size(); ) {
        char32_t character = next<replacement>(source, sourceOffset);
        if (character == sentinelCodePoint) {
            resultCode = ConversionResultCode::SourceInvalid;
            break;
        }
        if (bufferOffset == buffer.size()) {
            resultCode = ConversionResultCode::TargetExhausted;
            break;
        }
        bool sawError = append<replacement>(buffer, bufferOffset, character);
        if (sawError) {
            resultCode = ConversionResultCode::TargetExhausted;
            break;
        }
        orAllData |= character;
    }
    return { resultCode, buffer.first(bufferOffset), isASCII(orAllData) };
}

ConversionResult<char8_t> convert(std::span<const char16_t> source, std::span<char8_t> buffer)
{
    return convertInternal(source, buffer);
}

ConversionResult<char16_t> convert(std::span<const char8_t> source, std::span<char16_t> buffer)
{
    return convertInternal(source, buffer);
}

ConversionResult<char8_t> convert(std::span<const LChar> source, std::span<char8_t> buffer)
{
    return convertInternal(source, buffer);
}

ConversionResult<char8_t> convertReplacingInvalidSequences(std::span<const char16_t> source, std::span<char8_t> buffer)
{
    return convertInternal<Replacement::ReplaceInvalidSequences>(source, buffer);
}

ConversionResult<char16_t> convertReplacingInvalidSequences(std::span<const char8_t> source, std::span<char16_t> buffer)
{
    return convertInternal<Replacement::ReplaceInvalidSequences>(source, buffer);
}

CheckedUTF8 checkUTF8(std::span<const char8_t> source)
{
    size_t lengthUTF16 = 0;
    char32_t orAllData = 0;
    size_t sourceOffset;
    for (sourceOffset = 0; sourceOffset < source.size(); ) {
        size_t nextSourceOffset = sourceOffset;
        char32_t character = next(source, nextSourceOffset);
        if (character == sentinelCodePoint)
            break;
        sourceOffset = nextSourceOffset;
        lengthUTF16 += U16_LENGTH(character);
        orAllData |= character;
    }
    return { source.first(sourceOffset), lengthUTF16, isASCII(orAllData) };
}

UTF16LengthWithHash computeUTF16LengthWithHash(std::span<const char8_t> source)
{
    StringHasher hasher;
    size_t lengthUTF16 = 0;
    for (size_t sourceOffset = 0; sourceOffset < source.size(); ) {
        char32_t character = next(source, sourceOffset);
        if (character == sentinelCodePoint)
            return { };
        if (U_IS_BMP(character)) {
            hasher.addCharacter(character);
            ++lengthUTF16;
        } else {
            hasher.addCharacter(U16_LEAD(character));
            hasher.addCharacter(U16_TRAIL(character));
            lengthUTF16 += 2;
        }
    }
    return { lengthUTF16, hasher.hashWithTop8BitsMasked() };
}

template<typename CharacterTypeA, typename CharacterTypeB> bool equalInternal(std::span<CharacterTypeA> a, std::span<CharacterTypeB> b)
{
    size_t offsetA = 0;
    size_t offsetB = 0;
    while (offsetA < a.size() && offsetB < b.size()) {
        if (next(a, offsetA) != next(b, offsetB))
            return false;
    }
    return offsetA == a.size() && offsetB == b.size();
}

bool equal(std::span<const UChar> a, std::span<const char8_t> b)
{
    return equalInternal(a, b);
}

bool equal(std::span<const LChar> a, std::span<const char8_t> b)
{
    return equalInternal(a, b);
}

} // namespace WTF::Unicode
