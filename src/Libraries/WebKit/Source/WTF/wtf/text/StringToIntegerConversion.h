/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 28, 2023.
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

#include <wtf/CheckedArithmetic.h>
#include <wtf/text/ParsingUtilities.h>
#include <wtf/text/StringView.h>

namespace WTF {

// The parseInteger function template may allow leading and trailing spaces as defined by isUnicodeCompatibleASCIIWhitespace, and, after the leading spaces, allows a single leading "+".
// The parseIntegerAllowingTrailingJunk function template is like parseInteger, but allows any characters after the integer.

// FIXME: Should we add a version that does not allow "+"?
// FIXME: Should we add a version that allows other definitions of spaces, like isASCIIWhitespace or isASCIIWhitespaceWithoutFF?

enum class ParseIntegerWhitespacePolicy : bool { Disallow, Allow };

template<typename IntegralType> std::optional<IntegralType> parseInteger(StringView, uint8_t base = 10, ParseIntegerWhitespacePolicy = ParseIntegerWhitespacePolicy::Allow);
template<typename IntegralType> std::optional<IntegralType> parseIntegerAllowingTrailingJunk(StringView, uint8_t base = 10);

enum class TrailingJunkPolicy : bool { Disallow, Allow };

template<typename IntegralType, typename CharacterType> std::optional<IntegralType> parseInteger(std::span<const CharacterType> data, uint8_t base, TrailingJunkPolicy policy, ParseIntegerWhitespacePolicy whitespacePolicy = ParseIntegerWhitespacePolicy::Allow)
{
    if (!data.data())
        return std::nullopt;

    if (whitespacePolicy == ParseIntegerWhitespacePolicy::Allow)
        skipWhile<isUnicodeCompatibleASCIIWhitespace>(data);

    bool isNegative = false;
    if (std::is_signed_v<IntegralType> && skipExactly(data, '-'))
        isNegative = true;
    else
        skipExactly(data, '+');

    auto isCharacterAllowedInBase = [] (auto character, auto base) {
        if (isASCIIDigit(character))
            return character - '0' < base;
        return toASCIILowerUnchecked(character) >= 'a' && toASCIILowerUnchecked(character) < 'a' + std::min(base - 10, 26);
    };

    if (!(!data.empty() && isCharacterAllowedInBase(data.front(), base)))
        return std::nullopt;

    Checked<IntegralType, RecordOverflow> value;
    do {
        auto c = consume(data);
        IntegralType digitValue = isASCIIDigit(c) ? c - '0' : toASCIILowerUnchecked(c) - 'a' + 10;
        value *= static_cast<IntegralType>(base);
        if (isNegative)
            value -= digitValue;
        else
            value += digitValue;
    } while (!data.empty() && isCharacterAllowedInBase(data.front(), base));

    if (UNLIKELY(value.hasOverflowed()))
        return std::nullopt;

    if (policy == TrailingJunkPolicy::Disallow) {
        if (whitespacePolicy == ParseIntegerWhitespacePolicy::Allow)
            skipWhile<isUnicodeCompatibleASCIIWhitespace>(data);
        if (!data.empty())
            return std::nullopt;
    }

    return value.value();
}

template<typename IntegralType> std::optional<IntegralType> parseInteger(StringView string, uint8_t base, ParseIntegerWhitespacePolicy whitespacePolicy)
{
    if (string.is8Bit())
        return parseInteger<IntegralType>(string.span8(), base, TrailingJunkPolicy::Disallow, whitespacePolicy);
    return parseInteger<IntegralType>(string.span16(), base, TrailingJunkPolicy::Disallow, whitespacePolicy);
}

template<typename IntegralType> std::optional<IntegralType> parseIntegerAllowingTrailingJunk(StringView string, uint8_t base)
{
    if (string.is8Bit())
        return parseInteger<IntegralType>(string.span8(), base, TrailingJunkPolicy::Allow);
    return parseInteger<IntegralType>(string.span16(), base, TrailingJunkPolicy::Allow);
}

}

using WTF::parseInteger;
using WTF::parseIntegerAllowingTrailingJunk;
