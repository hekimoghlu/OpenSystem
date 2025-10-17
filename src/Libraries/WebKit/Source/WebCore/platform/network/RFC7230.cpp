/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 20, 2023.
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
#include "RFC7230.h"

#include <wtf/ASCIICType.h>
#include <wtf/text/StringView.h>

namespace RFC7230 {

bool isTokenCharacter(UChar c)
{
    return isASCIIAlpha(c) || isASCIIDigit(c)
        || c == '!' || c == '#' || c == '$'
        || c == '%' || c == '&' || c == '\''
        || c == '*' || c == '+' || c == '-'
        || c == '.' || c == '^' || c == '_'
        || c == '`' || c == '|' || c == '~';
}

bool isDelimiter(UChar c)
{
    return c == '(' || c == ')' || c == ','
        || c == '/' || c == ':' || c == ';'
        || c == '<' || c == '=' || c == '>'
        || c == '?' || c == '@' || c == '['
        || c == '\\' || c == ']' || c == '{'
        || c == '}' || c == '"';
}

static bool isVisibleCharacter(UChar c)
{
    return isTokenCharacter(c) || isDelimiter(c);
}

template<size_t min, size_t max>
static bool isInRange(UChar c)
{
    return c >= min && c <= max;
}

static bool isOBSText(UChar c)
{
    return isInRange<0x80, 0xFF>(c);
}

static bool isQuotedTextCharacter(UChar c)
{
    return isTabOrSpace(c)
        || c == 0x21
        || isInRange<0x23, 0x5B>(c)
        || isInRange<0x5D, 0x7E>(c)
        || isOBSText(c);
}

bool isQuotedPairSecondOctet(UChar c)
{
    return isTabOrSpace(c)
        || isVisibleCharacter(c)
        || isOBSText(c);
}

bool isCommentText(UChar c)
{
    return isTabOrSpace(c)
        || isInRange<0x21, 0x27>(c)
        || isInRange<0x2A, 0x5B>(c)
        || isInRange<0x5D, 0x7E>(c)
        || isOBSText(c);
}

bool isValidName(StringView name)
{
    if (!name.length())
        return false;
    for (size_t i = 0; i < name.length(); ++i) {
        if (!isTokenCharacter(name[i]))
            return false;
    }
    return true;
}

bool isValidValue(StringView value)
{
    enum class State {
        OptionalWhitespace,
        Token,
        QuotedString,
        Comment,
    };
    State state = State::OptionalWhitespace;
    size_t commentDepth = 0;
    bool hadNonWhitespace = false;

    for (size_t i = 0; i < value.length(); ++i) {
        UChar c = value[i];
        switch (state) {
        case State::OptionalWhitespace:
            if (isTabOrSpace(c))
                continue;
            hadNonWhitespace = true;
            if (isTokenCharacter(c)) {
                state = State::Token;
                continue;
            }
            if (c == '"') {
                state = State::QuotedString;
                continue;
            }
            if (c == '(') {
                ASSERT(!commentDepth);
                ++commentDepth;
                state = State::Comment;
                continue;
            }
            return false;

        case State::Token:
            if (isTokenCharacter(c))
                continue;
            state = State::OptionalWhitespace;
            continue;
        case State::QuotedString:
            if (c == '"') {
                state = State::OptionalWhitespace;
                continue;
            }
            if (c == '\\') {
                ++i;
                if (i == value.length())
                    return false;
                if (!isQuotedPairSecondOctet(value[i]))
                    return false;
                continue;
            }
            if (!isQuotedTextCharacter(c))
                return false;
            continue;
        case State::Comment:
            if (c == '(') {
                ++commentDepth;
                continue;
            }
            if (c == ')') {
                --commentDepth;
                if (!commentDepth)
                    state = State::OptionalWhitespace;
                continue;
            }
            if (c == '\\') {
                ++i;
                if (i == value.length())
                    return false;
                if (!isQuotedPairSecondOctet(value[i]))
                    return false;
                continue;
            }
            if (!isCommentText(c))
                return false;
            continue;
        }
    }

    switch (state) {
    case State::OptionalWhitespace:
    case State::Token:
        return hadNonWhitespace;
    case State::QuotedString:
    case State::Comment:
        // Unclosed comments or quotes are invalid values.
        break;
    }
    return false;
}

} // namespace RFC7230
