/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 19, 2023.
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

#include <wtf/text/WTFString.h>

namespace WebCore {

// Parses header fields into tokens, quoted strings and separators.
// Commonly used by ParsedContent* classes.
class HeaderFieldTokenizer {

public:
    explicit HeaderFieldTokenizer(const String&);

    // Try to parse a separator character, a token or either a token or a quoted
    // string from the |header_field| input. Return |true| on success. Return
    // |false| if the separator character, the token or the quoted string is
    // missing or invalid.
    bool consume(UChar);
    String consumeToken();
    String consumeTokenOrQuotedString();

    // Consume all characters before (but excluding) any of the characters from
    // the Vector parameter are found.
    // Because we potentially have to iterate through the entire Vector for each
    // character of the base string, the Vector should be small (< 3 members).
    void consumeBeforeAnyCharMatch(const Vector<UChar>&);

    bool isConsumed() const { return m_index >= m_input.length(); }

private:
    String consumeQuotedString();
    void skipSpaces();

    unsigned m_index = 0;
    const String m_input;
};

} // namespace WebCore
