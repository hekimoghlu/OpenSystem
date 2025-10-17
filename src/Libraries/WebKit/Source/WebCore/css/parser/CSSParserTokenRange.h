/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 8, 2022.
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
// Copyright (C) 2016 Apple Inc. All rights reserved.
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
#include "CSSTokenizer.h"
#include <wtf/Forward.h>
#include <wtf/text/ParsingUtilities.h>

namespace WebCore {

class StyleSheetContents;

// A CSSParserTokenRange is an iterator over a subrange of a vector of CSSParserTokens.
// Accessing outside of the range will return an endless stream of EOF tokens.
// This class refers to half-open intervals [first, last).
class CSSParserTokenRange {
public:
    CSSParserTokenRange() = default;

    template<size_t inlineBuffer>
    CSSParserTokenRange(const Vector<CSSParserToken, inlineBuffer>& vector)
        : m_tokens(vector.span())
    {
    }

    // This should be called on a range with tokens returned by that range.
    CSSParserTokenRange makeSubRange(const CSSParserToken* first, const CSSParserToken* last) const;
    CSSParserTokenRange makeSubRange(std::span<const CSSParserToken> subrange) const;

    bool atEnd() const { return m_tokens.empty(); }

    const CSSParserToken* begin() const { return std::to_address(m_tokens.begin()); }
    const CSSParserToken* end() const { return std::to_address(m_tokens.end()); }

    size_t size() const { return m_tokens.size(); }

    const CSSParserToken& peek(size_t offset = 0) const
    {
        if (offset < m_tokens.size())
            return m_tokens[offset];
        return eofToken();
    }

    const CSSParserToken& consume()
    {
        if (m_tokens.empty())
            return eofToken();
        return WTF::consume(m_tokens);
    }

    const CSSParserToken& consumeIncludingWhitespace()
    {
        auto& result = consume();
        consumeWhitespace();
        return result;
    }

    // The returned range doesn't include the brackets
    CSSParserTokenRange consumeBlock();
    CSSParserTokenRange consumeBlockCheckingForEditability(StyleSheetContents*);

    void consumeComponentValue();

    void consumeWhitespace()
    {
        size_t i = 0;
        for (; i < m_tokens.size() && CSSTokenizer::isWhitespace(m_tokens[i].type()); ++i) { }
        skip(m_tokens, i);
    }

    void trimTrailingWhitespace();
    const CSSParserToken& consumeLast();

    CSSParserTokenRange consumeAll() { return { std::exchange(m_tokens, std::span<const CSSParserToken> { }) }; }

    String serialize(CSSParserToken::SerializationMode = CSSParserToken::SerializationMode::Normal) const;

    std::span<const CSSParserToken> span() const { return m_tokens; }

    static CSSParserToken& eofToken();

private:
    CSSParserTokenRange(std::span<const CSSParserToken> tokens)
        : m_tokens(tokens)
    { }

    std::span<const CSSParserToken> m_tokens;
};

} // namespace WebCore
