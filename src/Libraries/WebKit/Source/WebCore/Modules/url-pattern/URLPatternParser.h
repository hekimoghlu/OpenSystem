/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 11, 2022.
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

#include "URLPatternTokenizer.h"
#include <wtf/text/StringBuilder.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

enum class EncodingCallbackType : uint8_t;
template<typename> class ExceptionOr;

namespace URLPatternUtilities {

struct Token;
enum class TokenType : uint8_t;

enum class PartType : uint8_t { FixedText, Regexp, SegmentWildcard, FullWildcard };
enum class Modifier : uint8_t { None, Optional, ZeroOrMore, OneOrMore };
enum class IsFirst : bool { No, Yes };

struct Part {
    PartType type;
    String value;
    Modifier modifier;
    String name { };
    String prefix { };
    String suffix { };
};

struct URLPatternStringOptions {
    String delimiterCodepoint { };
    String prefixCodepoint { };
    bool ignoreCase { false };
};

class URLPatternParser {
public:
    URLPatternParser(EncodingCallbackType, String&& segmentWildcardRegexp);
    ExceptionOr<void> performParse(const URLPatternStringOptions&);

    void setTokenList(Vector<Token>&& tokenList) { m_tokenList = WTFMove(tokenList); }
    static ExceptionOr<Vector<Part>> parse(StringView, const URLPatternStringOptions&, EncodingCallbackType);

private:
    Token tryToConsumeToken(TokenType);
    Token tryToConsumeRegexOrWildcardToken(const Token&);
    Token tryToConsumeModifierToken();

    String consumeText();
    ExceptionOr<Token> consumeRequiredToken(TokenType);

    ExceptionOr<void> maybeAddPartFromPendingFixedValue();
    ExceptionOr<void> addPart(String&& prefix, const Token& nameToken, const Token& regexpOrWildcardToken, String&& suffix, const Token& modifierToken);

    bool isDuplicateName(StringView) const;

    Vector<Part> takePartList() { return std::exchange(m_partList, { }); }

    Vector<Token> m_tokenList;
    Vector<Part> m_partList;
    EncodingCallbackType m_callbackType;
    String m_segmentWildcardRegexp;
    StringBuilder m_pendingFixedValue;
    size_t m_index { 0 };
    int m_nextNumericName { 0 };
};

// FIXME: Consider moving functions to somewhere generic, perhaps refactor Part to its own class.
String generateSegmentWildcardRegexp(const URLPatternStringOptions&);
String escapeRegexString(StringView);
ASCIILiteral convertModifierToString(Modifier);
std::pair<String, Vector<String>> generateRegexAndNameList(const Vector<Part>& partList, const URLPatternStringOptions&);
String generatePatternString(const Vector<Part>& partList, const URLPatternStringOptions&);
String escapePatternString(StringView input);
bool isValidNameCodepoint(UChar codepoint, URLPatternUtilities::IsFirst);


} // namespace URLPatternUtilities
} // namespace WebCore
