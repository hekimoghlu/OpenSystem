/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 10, 2023.
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

#include "Token.h"
#include <wtf/ASCIICType.h>
#include <wtf/text/StringParsingBuffer.h>
#include <wtf/text/WTFString.h>

namespace WGSL {

template<typename T>
class Lexer {
public:
    Lexer(const String& wgsl)
    {
        if constexpr (std::is_same<T, LChar>::value)
            m_code = wgsl.span8();
        else {
            static_assert(std::is_same<T, UChar>::value, "The lexer expects its template parameter to be either LChar or UChar");
            m_code = wgsl.span16();
            ASSERT(!(wgsl.sizeInBytes() % 2));
        }

        m_current = m_code.hasCharactersRemaining() ? m_code[0] : 0;
        m_currentPosition = { 1, 0, 0 };
    }

    Vector<Token> lex();
    bool isAtEndOfFile() const;

private:
    Token nextToken();
    Token lexNumber();
    unsigned currentOffset() const { return m_currentPosition.offset; }
    unsigned currentTokenLength() const { return currentOffset() - m_tokenStartingPosition.offset; }

    Token makeToken(TokenType type)
    {
        return { type, m_tokenStartingPosition, currentTokenLength() };
    }
    Token makeFloatToken(TokenType type, double floatValue)
    {
        return { type, m_tokenStartingPosition, currentTokenLength(), floatValue };
    }

    Token makeIntegerToken(TokenType type, int64_t integerValue)
    {
        return { type, m_tokenStartingPosition, currentTokenLength(), integerValue };
    }

    Token makeIdentifierToken(String&& identifier)
    {
        return { WGSL::TokenType::Identifier, m_tokenStartingPosition, currentTokenLength(), WTFMove(identifier) };
    }

    T shift(unsigned = 1);
    T peek(unsigned = 0);
    void newLine();
    bool skipBlockComments();
    void skipLineComment();
    bool skipWhitespaceAndComments();

    T m_current;
    StringParsingBuffer<T> m_code;
    SourcePosition m_currentPosition { 0, 0, 0 };
    SourcePosition m_tokenStartingPosition { 0, 0, 0 };
};

}
