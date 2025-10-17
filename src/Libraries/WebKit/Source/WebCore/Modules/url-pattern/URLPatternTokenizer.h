/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 17, 2022.
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

#include "ExceptionOr.h"
#include <wtf/text/StringView.h>

namespace WebCore {
namespace URLPatternUtilities {

enum class TokenType : uint8_t { Open, Close, Regexp, Name, Char, EscapedChar, OtherModifier, Asterisk, End, InvalidChar };
enum class TokenizePolicy : bool { Strict, Lenient };

struct Token {
    TokenType type;
    std::optional<size_t> index;
    StringView value;

    bool isNull() const;
};

class Tokenizer {
public:
    Tokenizer(StringView input, TokenizePolicy tokenizerPolicy);

    ExceptionOr<Vector<Token>> tokenize();

private:
    StringView m_input;
    TokenizePolicy m_policy { TokenizePolicy::Strict };
    Vector<Token> m_tokenList;
    size_t m_index { 0 };
    size_t m_nextIndex { 0 };
    char32_t m_codepoint;

    void getNextCodePoint();
    void seekNextCodePoint(size_t index);

    void addToken(TokenType currentType, size_t nextPosition, size_t valuePosition, size_t valueLength);
    void addToken(TokenType currentType, size_t nextPosition, size_t valuePosition);
    void addToken(TokenType currentType);

    ExceptionOr<void> processTokenizingError(size_t nextPosition, size_t valuePosition, const String&);
};

} // namespace URLPatternUtilities
} // namespace WebCore
