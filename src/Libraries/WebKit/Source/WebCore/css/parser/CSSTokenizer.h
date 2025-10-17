/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 19, 2023.
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

// Copyright 2014 The Chromium Authors. All rights reserved.
// Copyright (C) 2016-2024 Apple Inc. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//    * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//    * Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
//    * Neither the name of Google Inc. nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#pragma once

#include "CSSParserToken.h"
#include "CSSTokenizerInputStream.h"
#include <climits>
#include <wtf/text/StringView.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

class CSSTokenizerInputStream;
class CSSParserObserverWrapper;
class CSSParserTokenRange;

DECLARE_ALLOCATOR_WITH_HEAP_IDENTIFIER(CSSTokenizer);
class CSSTokenizer {
    WTF_MAKE_NONCOPYABLE(CSSTokenizer);
    WTF_MAKE_FAST_ALLOCATED_WITH_HEAP_IDENTIFIER(CSSTokenizer);
public:
    static std::unique_ptr<CSSTokenizer> tryCreate(const String&);
    static std::unique_ptr<CSSTokenizer> tryCreate(const String&, CSSParserObserverWrapper&); // For the inspector

    WEBCORE_EXPORT explicit CSSTokenizer(const String&);
    CSSTokenizer(const String&, CSSParserObserverWrapper&); // For the inspector

    WEBCORE_EXPORT CSSParserTokenRange tokenRange() const;
    unsigned tokenCount();

    static bool isWhitespace(CSSParserTokenType);
    static bool isNewline(UChar);

    Vector<String>&& escapedStringsForAdoption() { return WTFMove(m_stringPool); }

private:
    CSSTokenizer(const String&, CSSParserObserverWrapper*, bool* constructionSuccess);

    CSSParserToken nextToken();

    UChar consume();
    void reconsume(UChar);

    String preprocessString(const String&);

    CSSParserToken consumeNumericToken();
    CSSParserToken consumeIdentLikeToken();
    CSSParserToken consumeNumber();
    CSSParserToken consumeStringTokenUntil(UChar);
    CSSParserToken consumeURLToken();

    void consumeBadUrlRemnants();
    void consumeSingleWhitespaceIfNext();
    void consumeUntilCommentEndFound();

    bool consumeIfNext(UChar);
    StringView consumeName();
    char32_t consumeEscape();

    bool nextTwoCharsAreValidEscape();
    bool nextCharsAreNumber(UChar);
    bool nextCharsAreNumber();
    bool nextCharsAreIdentifier(UChar);
    bool nextCharsAreIdentifier();

    CSSParserToken blockStart(CSSParserTokenType);
    CSSParserToken blockStart(CSSParserTokenType blockType, CSSParserTokenType, StringView);
    CSSParserToken blockEnd(CSSParserTokenType, CSSParserTokenType startType);

    CSSParserToken newline(UChar);
    CSSParserToken whitespace(UChar);
    CSSParserToken leftParenthesis(UChar);
    CSSParserToken rightParenthesis(UChar);
    CSSParserToken leftBracket(UChar);
    CSSParserToken rightBracket(UChar);
    CSSParserToken leftBrace(UChar);
    CSSParserToken rightBrace(UChar);
    CSSParserToken plusOrFullStop(UChar);
    CSSParserToken comma(UChar);
    CSSParserToken hyphenMinus(UChar);
    CSSParserToken asterisk(UChar);
    CSSParserToken lessThan(UChar);
    CSSParserToken solidus(UChar);
    CSSParserToken colon(UChar);
    CSSParserToken semiColon(UChar);
    CSSParserToken hash(UChar);
    CSSParserToken circumflexAccent(UChar);
    CSSParserToken dollarSign(UChar);
    CSSParserToken verticalLine(UChar);
    CSSParserToken tilde(UChar);
    CSSParserToken commercialAt(UChar);
    CSSParserToken reverseSolidus(UChar);
    CSSParserToken asciiDigit(UChar);
    CSSParserToken letterU(UChar);
    CSSParserToken nameStart(UChar);
    CSSParserToken stringStart(UChar);
    CSSParserToken endOfFile(UChar);

    StringView registerString(const String&);

    using CodePoint = CSSParserToken (CSSTokenizer::*)(UChar);
    static const std::array<CodePoint, 128> codePoints;

    Vector<CSSParserTokenType, 8> m_blockStack;
    Vector<CSSParserToken, 32> m_tokens;
    // We only allocate strings when escapes are used.
    Vector<String> m_stringPool;
    CSSTokenizerInputStream m_input;
};

} // namespace WebCore
