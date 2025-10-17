/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 19, 2023.
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
#include "ScriptExecutionContext.h"
#include "URLPatternInit.h"

namespace WebCore {

enum class EncodingCallbackType : uint8_t;

namespace URLPatternUtilities {
struct Token;
enum class TokenType : uint8_t;
struct URLPatternStringOptions;
struct URLPatternInit;
}

enum class URLPatternConstructorStringParserState : uint8_t { Init, Protocol, Authority, Username, Password, Hostname, Port, Pathname, Search, Hash, Done };

class URLPatternConstructorStringParser {
public:
    explicit URLPatternConstructorStringParser(String&& input);
    ExceptionOr<URLPatternInit> parse(ScriptExecutionContext&);

private:
    void performParse(ScriptExecutionContext&);
    void rewind();
    const URLPatternUtilities::Token& getSafeToken(size_t index) const;
    bool isNonSpecialPatternChararacter(size_t index, char value) const;
    bool isSearchPrefix() const;
    bool isAuthoritySlashesNext() const;
    String makeComponentString() const;
    void changeState(URLPatternConstructorStringParserState, size_t skip);
    void updateState(ScriptExecutionContext&);
    ExceptionOr<void> computeProtocolMatchSpecialSchemeFlag(ScriptExecutionContext&);

    StringView m_input;
    Vector<URLPatternUtilities::Token> m_tokenList;
    URLPatternInit m_result;
    size_t m_componentStart { 0 };
    size_t m_tokenIndex { 0 };
    size_t m_tokenIncrement { 1 };
    size_t m_groupDepth { 0 };
    int m_hostnameIPv6BracketDepth { 0 };
    bool m_protocolMatchesSpecialSchemeFlag { false };
    URLPatternConstructorStringParserState m_state { URLPatternConstructorStringParserState::Init };

};

}
