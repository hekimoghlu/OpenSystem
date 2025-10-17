/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 13, 2022.
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
#include "HeaderFieldTokenizer.h"

#include "RFC7230.h"
#include <wtf/text/StringBuilder.h>

namespace WebCore {

HeaderFieldTokenizer::HeaderFieldTokenizer(const String& headerField)
    : m_input(headerField)
{
    skipSpaces();
}

bool HeaderFieldTokenizer::consume(UChar c)
{
    ASSERT(!isTabOrSpace(c));

    if (isConsumed() || m_input[m_index] != c)
        return false;

    ++m_index;
    skipSpaces();
    return true;
}

String HeaderFieldTokenizer::consumeQuotedString()
{
    StringBuilder builder;

    ASSERT(m_input[m_index] == '"');
    ++m_index;

    while (!isConsumed()) {
        if (m_input[m_index] == '"') {
            String output = builder.toString();
            ++m_index;
            skipSpaces();
            return output;
        }
        if (m_input[m_index] == '\\') {
            ++m_index;
            if (isConsumed())
                return String();
        }
        builder.append(m_input[m_index]);
        ++m_index;
    }
    return String();
}

String HeaderFieldTokenizer::consumeToken()
{
    auto start = m_index;
    while (!isConsumed() && RFC7230::isTokenCharacter(m_input[m_index]))
        ++m_index;

    if (start == m_index)
        return String();

    String output = m_input.substring(start, m_index - start);
    skipSpaces();
    return output;
}

String HeaderFieldTokenizer::consumeTokenOrQuotedString()
{
    if (isConsumed())
        return String();

    if (m_input[m_index] == '"')
        return consumeQuotedString();

    return consumeToken();
}

void HeaderFieldTokenizer::skipSpaces()
{
    while (!isConsumed() && isTabOrSpace(m_input[m_index]))
        ++m_index;
}

void HeaderFieldTokenizer::consumeBeforeAnyCharMatch(const Vector<UChar>& chars)
{
    ASSERT(chars.size() > 0U && chars.size() < 3U);

    while (!isConsumed()) {
        for (const auto& c : chars) {
            if (c == m_input[m_index])
                return;
        }

        ++m_index;
    }
}

} // namespace WebCore
