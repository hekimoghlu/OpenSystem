/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 10, 2022.
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

#include <wtf/Forward.h>
#include <wtf/text/ParsingUtilities.h>

typedef std::pair<char32_t, char32_t> UnicodeRange;
typedef Vector<UnicodeRange> UnicodeRanges;

namespace WebCore {

class FloatPoint;
class FloatRect;

enum class SuffixSkippingPolicy {
    DontSkip,
    Skip
};

std::optional<float> parseNumber(StringParsingBuffer<LChar>&, SuffixSkippingPolicy = SuffixSkippingPolicy::Skip);
std::optional<float> parseNumber(StringParsingBuffer<UChar>&, SuffixSkippingPolicy = SuffixSkippingPolicy::Skip);
std::optional<float> parseNumber(StringView, SuffixSkippingPolicy = SuffixSkippingPolicy::Skip);

std::optional<std::pair<float, float>> parseNumberOptionalNumber(StringView);

std::optional<bool> parseArcFlag(StringParsingBuffer<LChar>&);
std::optional<bool> parseArcFlag(StringParsingBuffer<UChar>&);

std::optional<FloatPoint> parsePoint(StringView);
std::optional<FloatRect> parseRect(StringView);

std::optional<FloatPoint> parseFloatPoint(StringParsingBuffer<LChar>&);
std::optional<FloatPoint> parseFloatPoint(StringParsingBuffer<UChar>&);

std::optional<std::pair<UnicodeRanges, UncheckedKeyHashSet<String>>> parseKerningUnicodeString(StringView);
std::optional<UncheckedKeyHashSet<String>> parseGlyphName(StringView);

template<typename CharacterType> constexpr bool isSVGSpaceOrComma(CharacterType c)
{
    return isASCIIWhitespace(c) || c == ',';
}

template<typename CharacterType> constexpr bool skipOptionalSVGSpaces(const CharacterType*& ptr, const CharacterType* end)
{
    skipWhile<isASCIIWhitespace>(ptr, end);
    return ptr < end;
}

template<typename CharacterType> constexpr bool skipOptionalSVGSpaces(StringParsingBuffer<CharacterType>& characters)
{
    skipWhile<isASCIIWhitespace>(characters);
    return characters.hasCharactersRemaining();
}

template<typename CharacterType> constexpr bool skipOptionalSVGSpacesOrDelimiter(const CharacterType*& ptr, const CharacterType* end, char delimiter = ',')
{
    if (ptr < end && !isASCIIWhitespace(*ptr) && *ptr != delimiter)
        return false;
    if (skipOptionalSVGSpaces(ptr, end)) {
        if (ptr < end && *ptr == delimiter) {
            ptr++;
            skipOptionalSVGSpaces(ptr, end);
        }
    }
    return ptr < end;
}

template<typename CharacterType> constexpr bool skipOptionalSVGSpacesOrDelimiter(StringParsingBuffer<CharacterType>& characters, char delimiter = ',')
{
    if (!characters.hasCharactersRemaining())
        return false;

    if (!isASCIIWhitespace(*characters) && *characters != delimiter)
        return false;

    // There are only spaces in the remaining characters.
    if (!skipOptionalSVGSpaces(characters))
        return false;

    if (*characters != delimiter)
        return true;

    // A delimiter is hit. Skip the following spaces also e.g. " , ".
    return skipOptionalSVGSpaces(++characters);
}

} // namespace WebCore
