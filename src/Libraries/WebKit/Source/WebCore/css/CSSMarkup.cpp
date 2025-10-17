/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 24, 2023.
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
#include "CSSMarkup.h"

#include "CSSParserIdioms.h"
#include <wtf/HexNumber.h>
#include <wtf/text/ParsingUtilities.h>
#include <wtf/text/StringBuilder.h>
#include <wtf/unicode/CharacterNames.h>

namespace WebCore {

template <typename CharacterType>
static inline bool isCSSTokenizerIdentifier(std::span<const CharacterType> characters)
{
    // -?
    skipWhile(characters, '-');

    // {nmstart}
    if (!skipExactly<isNameStartCodePoint>(characters))
        return false;

    // {nmchar}*
    skipWhile<isNameCodePoint>(characters);

    return characters.empty();
}

// "ident" from the CSS tokenizer, minus backslash-escape sequences
static bool isCSSTokenizerIdentifier(const String& string)
{
    if (string.isEmpty())
        return false;

    if (string.is8Bit())
        return isCSSTokenizerIdentifier(string.span8());
    return isCSSTokenizerIdentifier(string.span16());
}

static void serializeCharacter(char32_t c, StringBuilder& appendTo)
{
    appendTo.append('\\', c);
}

static void serializeCharacterAsCodePoint(char32_t c, StringBuilder& appendTo)
{
    appendTo.append('\\', hex(c, Lowercase), ' ');
}

void serializeIdentifier(const String& identifier, StringBuilder& appendTo, bool skipStartChecks)
{
    bool isFirst = !skipStartChecks;
    bool isSecond = false;
    bool isFirstCharHyphen = false;
    unsigned index = 0;
    while (index < identifier.length()) {
        char32_t c = identifier.characterStartingAt(index);
        if (!c) {
            // Check for lone surrogate which characterStartingAt does not return.
            c = identifier[index];
        }

        index += U16_LENGTH(c);

        if (!c)
            appendTo.append(replacementCharacter);
        else if (c <= 0x1f || c == deleteCharacter || (0x30 <= c && c <= 0x39 && (isFirst || (isSecond && isFirstCharHyphen))))
            serializeCharacterAsCodePoint(c, appendTo);
        else if (c == hyphenMinus && isFirst && index == identifier.length())
            serializeCharacter(c, appendTo);
        else if (0x80 <= c || c == hyphenMinus || c == lowLine || (0x30 <= c && c <= 0x39) || (0x41 <= c && c <= 0x5a) || (0x61 <= c && c <= 0x7a))
            appendTo.append(c);
        else
            serializeCharacter(c, appendTo);

        if (isFirst) {
            isFirst = false;
            isSecond = true;
            isFirstCharHyphen = (c == hyphenMinus);
        } else if (isSecond)
            isSecond = false;
    }
}

void serializeString(const String& string, StringBuilder& appendTo)
{
    appendTo.append('"');

    unsigned index = 0;
    while (index < string.length()) {
        char32_t c = string.characterStartingAt(index);
        index += U16_LENGTH(c);

        if (c <= 0x1f || c == deleteCharacter)
            serializeCharacterAsCodePoint(c, appendTo);
        else if (c == quotationMark || c == reverseSolidus)
            serializeCharacter(c, appendTo);
        else
            appendTo.append(c);
    }

    appendTo.append('"');
}

String serializeString(const String& string)
{
    StringBuilder builder;
    serializeString(string, builder);
    return builder.toString();
}

String serializeURL(const String& string)
{
    StringBuilder builder;
    builder.append("url("_s);
    serializeString(string, builder);
    builder.append(')');
    return builder.toString();
}

String serializeFontFamily(const String& string)
{
    return isCSSTokenizerIdentifier(string) ? string : serializeString(string);
}

} // namespace WebCore
