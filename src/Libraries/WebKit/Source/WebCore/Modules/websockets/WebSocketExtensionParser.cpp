/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 15, 2023.
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

#include "WebSocketExtensionParser.h"

#include <wtf/ASCIICType.h>
#include <wtf/text/CString.h>
#include <wtf/text/ParsingUtilities.h>

namespace WebCore {

bool WebSocketExtensionParser::finished()
{
    return m_data.empty();
}

bool WebSocketExtensionParser::parsedSuccessfully()
{
    return m_data.empty() && !m_didFailParsing;
}

static bool isSeparator(char character)
{
    static constexpr auto separatorCharacters = "()<>@,;:\\\"/[]?={} \t"_span;
    return contains(separatorCharacters, character);
}

static bool isSpaceOrTab(LChar character)
{
    return character == ' ' || character == '\t';
}

void WebSocketExtensionParser::skipSpaces()
{
    skipWhile<isSpaceOrTab>(m_data);
}

bool WebSocketExtensionParser::consumeToken()
{
    skipSpaces();
    auto start = m_data;
    size_t tokenLength = 0;
    while (tokenLength < m_data.size() && isASCIIPrintable(m_data[tokenLength]) && !isSeparator(m_data[tokenLength]))
        ++tokenLength;
    if (tokenLength) {
        m_currentToken = String(consumeSpan(start, tokenLength));
        return true;
    }
    return false;
}

bool WebSocketExtensionParser::consumeQuotedString()
{
    skipSpaces();
    if (!skipExactly(m_data, '"'))
        return false;

    Vector<char> buffer;
    while (!m_data.empty() && m_data[0] != '"') {
        if (skipExactly(m_data, '\\')) {
            if (m_data.empty())
                return false;
        }
        buffer.append(consume(m_data));
    }
    if (m_data.empty() || m_data[0] != '"')
        return false;
    m_currentToken = String::fromUTF8(buffer.span());
    if (m_currentToken.isNull())
        return false;
    skip(m_data, 1);
    return true;
}

bool WebSocketExtensionParser::consumeQuotedStringOrToken()
{
    // This is ok because consumeQuotedString() doesn't update m_data or
    // set m_didFailParsing to true on failure.
    return consumeQuotedString() || consumeToken();
}

bool WebSocketExtensionParser::consumeCharacter(char character)
{
    skipSpaces();
    return skipExactly(m_data, character);
}

bool WebSocketExtensionParser::parseExtension(String& extensionToken, HashMap<String, String>& extensionParameters)
{
    // Parse extension-token.
    if (!consumeToken()) {
        m_didFailParsing = true;
        return false;
    }

    extensionToken = currentToken();

    // Parse extension-parameters if exists.
    while (consumeCharacter(';')) {
        if (!consumeToken()) {
            m_didFailParsing = true;
            return false;
        }

        String parameterToken = currentToken();
        if (consumeCharacter('=')) {
            if (consumeQuotedStringOrToken())
                extensionParameters.add(parameterToken, currentToken());
            else {
                m_didFailParsing = true;
                return false;
            }
        } else
            extensionParameters.add(parameterToken, String());
    }
    if (!finished() && !consumeCharacter(',')) {
        m_didFailParsing = true;
        return false;
    }

    return true;
}

} // namespace WebCore
