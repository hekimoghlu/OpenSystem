/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 19, 2025.
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
#include "IDBKeyPath.h"

#include <wtf/ASCIICType.h>
#include <wtf/dtoa.h>
#include <wtf/text/MakeString.h>
#include <wtf/text/StringBuilder.h>

namespace WebCore {

class IDBKeyPathLexer {
public:
    enum TokenType {
        TokenIdentifier,
        TokenDot,
        TokenEnd,
        TokenError
    };

    explicit IDBKeyPathLexer(const String& s)
        : m_string(s)
        , m_remainingText(s)
        , m_currentTokenType(TokenError)
    {
    }

    TokenType currentTokenType() const { return m_currentTokenType; }

    TokenType nextTokenType()
    {
        m_currentTokenType = lex(m_currentElement);
        return m_currentTokenType;
    }

    const String& currentElement() { return m_currentElement; }

private:
    TokenType lex(String&);
    TokenType lexIdentifier(String&);

    String m_currentElement;
    const String m_string;
    StringView m_remainingText;
    TokenType m_currentTokenType;
};

IDBKeyPathLexer::TokenType IDBKeyPathLexer::lex(String& element)
{
    if (m_remainingText.isEmpty())
        return TokenEnd;

    if (m_remainingText[0] == '.') {
        m_remainingText = m_remainingText.substring(1);
        return TokenDot;
    }

    return lexIdentifier(element);
}

namespace {

// The following correspond to grammar in ECMA-262.
const uint32_t unicodeLetter = U_GC_L_MASK | U_GC_NL_MASK;
const uint32_t unicodeCombiningMark = U_GC_MN_MASK | U_GC_MC_MASK;
const uint32_t unicodeDigit = U_GC_ND_MASK;
const uint32_t unicodeConnectorPunctuation = U_GC_PC_MASK;
const UChar ZWNJ = 0x200C;
const UChar ZWJ = 0x200D;

static inline bool isIdentifierStartCharacter(UChar c)
{
    return (U_GET_GC_MASK(c) & unicodeLetter) || (c == '$') || (c == '_');
}

static inline bool isIdentifierCharacter(UChar c)
{
    return (U_GET_GC_MASK(c) & (unicodeLetter | unicodeCombiningMark | unicodeDigit | unicodeConnectorPunctuation)) || (c == '$') || (c == '_') || (c == ZWNJ) || (c == ZWJ);
}

} // namespace

IDBKeyPathLexer::TokenType IDBKeyPathLexer::lexIdentifier(String& element)
{
    StringView start = m_remainingText;
    if (!m_remainingText.isEmpty() && isIdentifierStartCharacter(m_remainingText[0]))
        m_remainingText = m_remainingText.substring(1);
    else
        return TokenError;

    while (!m_remainingText.isEmpty() && isIdentifierCharacter(m_remainingText[0]))
        m_remainingText = m_remainingText.substring(1);

    element = start.left(start.length() - m_remainingText.length()).toString();
    return TokenIdentifier;
}

static bool IDBIsValidKeyPath(const String& keyPath)
{
    IDBKeyPathParseError error;
    Vector<String> keyPathElements;
    IDBParseKeyPath(keyPath, keyPathElements, error);
    return error == IDBKeyPathParseError::None;
}

void IDBParseKeyPath(const String& keyPath, Vector<String>& elements, IDBKeyPathParseError& error)
{
    // IDBKeyPath ::= EMPTY_STRING | identifier ('.' identifier)*
    // The basic state machine is:
    //   Start => {Identifier, End}
    //   Identifier => {Dot, End}
    //   Dot => {Identifier}
    // It bails out as soon as it finds an error, but doesn't discard the bits it managed to parse.
    enum ParserState { Identifier, Dot, End };

    IDBKeyPathLexer lexer(keyPath);
    IDBKeyPathLexer::TokenType tokenType = lexer.nextTokenType();
    ParserState state;
    if (tokenType == IDBKeyPathLexer::TokenIdentifier)
        state = Identifier;
    else if (tokenType == IDBKeyPathLexer::TokenEnd)
        state = End;
    else {
        error = IDBKeyPathParseError::Start;
        return;
    }

    while (1) {
        switch (state) {
        case Identifier : {
            IDBKeyPathLexer::TokenType tokenType = lexer.currentTokenType();
            ASSERT(tokenType == IDBKeyPathLexer::TokenIdentifier);

            String element = lexer.currentElement();
            elements.append(element);

            tokenType = lexer.nextTokenType();
            if (tokenType == IDBKeyPathLexer::TokenDot)
                state = Dot;
            else if (tokenType == IDBKeyPathLexer::TokenEnd)
                state = End;
            else {
                error = IDBKeyPathParseError::Identifier;
                return;
            }
            break;
        }
        case Dot: {
            IDBKeyPathLexer::TokenType tokenType = lexer.currentTokenType();
            ASSERT(tokenType == IDBKeyPathLexer::TokenDot);

            tokenType = lexer.nextTokenType();
            if (tokenType == IDBKeyPathLexer::TokenIdentifier)
                state = Identifier;
            else {
                error = IDBKeyPathParseError::Dot;
                return;
            }
            break;
        }
        case End: {
            error = IDBKeyPathParseError::None;
            return;
        }
        }
    }
}

bool isIDBKeyPathValid(const IDBKeyPath& keyPath)
{
    auto visitor = WTF::makeVisitor([](const String& string) {
        return IDBIsValidKeyPath(string);
    }, [](const Vector<String>& vector) {
        if (vector.isEmpty())
            return false;
        for (auto& key : vector) {
            if (!IDBIsValidKeyPath(key))
                return false;
        }
        return true;
    });
    return std::visit(visitor, keyPath);
}

#if !LOG_DISABLED
String loggingString(const IDBKeyPath& path)
{
    auto visitor = WTF::makeVisitor([](const String& string) {
        return makeString("< "_s, string, " >"_s);
    }, [](const Vector<String>& strings) {
        if (strings.isEmpty())
            return "< >"_str;

        StringBuilder builder;
        builder.append("< "_s);
        for (size_t i = 0; i < strings.size() - 1; ++i)
            builder.append(strings[i], ", "_s);
        builder.append(strings.last(), " >"_s);

        return builder.toString();
    });

    return std::visit(visitor, path);
}
#endif

} // namespace WebCore
