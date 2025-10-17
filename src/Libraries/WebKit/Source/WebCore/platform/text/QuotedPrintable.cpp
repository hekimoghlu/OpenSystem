/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 8, 2024.
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
#include "QuotedPrintable.h"

#include <wtf/ASCIICType.h>
#include <wtf/Vector.h>
#include <wtf/text/ASCIILiteral.h>

namespace WebCore {

static const size_t maximumLineLength = 76;

static constexpr auto crlfLineEnding = "\r\n"_s;

static size_t lengthOfLineEndingAtIndex(std::span<const uint8_t> input, size_t index)
{
    ASSERT_WITH_SECURITY_IMPLICATION(index < input.size());
    if (input[index] == '\n')
        return 1; // Single LF.

    if (input[index] == '\r') {
        if ((index + 1) == input.size() || input[index + 1] != '\n')
            return 1; // Single CR (Classic Mac OS).
        return 2; // CR-LF.
    }

    return 0;
}

Vector<uint8_t> quotedPrintableEncode(const Vector<uint8_t>& input)
{
    return quotedPrintableEncode(input.span());
}

Vector<uint8_t> quotedPrintableEncode(std::span<const uint8_t> input)
{
    Vector<uint8_t> out;
    out.reserveInitialCapacity(input.size());
    size_t currentLineLength = 0;
    for (size_t i = 0; i < input.size(); ++i) {
        bool isLastCharacter = (i == input.size() - 1);
        uint8_t currentCharacter = input[i];
        bool requiresEncoding = false;
        // All non-printable ASCII characters and = require encoding.
        if ((currentCharacter < ' ' || currentCharacter > '~' || currentCharacter == '=') && currentCharacter != '\t')
            requiresEncoding = true;

        // Space and tab characters have to be encoded if they appear at the end of a line.
        if (!requiresEncoding && (currentCharacter == '\t' || currentCharacter == ' ') && (isLastCharacter || lengthOfLineEndingAtIndex(input, i + 1)))
            requiresEncoding = true;

        // End of line should be converted to CR-LF sequences.
        if (!isLastCharacter) {
            size_t lengthOfLineEnding = lengthOfLineEndingAtIndex(input, i);
            if (lengthOfLineEnding) {
                out.append(crlfLineEnding.span8());
                currentLineLength = 0;
                i += (lengthOfLineEnding - 1); // -1 because we'll ++ in the for() above.
                continue;
            }
        }

        size_t lengthOfEncodedCharacter = 1;
        if (requiresEncoding)
            lengthOfEncodedCharacter += 2;
        if (!isLastCharacter)
            lengthOfEncodedCharacter += 1; // + 1 for the = (soft line break).

        // Insert a soft line break if necessary.
        if (currentLineLength + lengthOfEncodedCharacter > maximumLineLength) {
            out.append('=');
            out.append(crlfLineEnding.span8());
            currentLineLength = 0;
        }

        // Finally, insert the actual character(s).
        if (requiresEncoding) {
            out.append('=');
            out.append(upperNibbleToASCIIHexDigit(currentCharacter));
            out.append(lowerNibbleToASCIIHexDigit(currentCharacter));
            currentLineLength += 3;
        } else {
            out.append(currentCharacter);
            currentLineLength++;
        }
    }

    return out;
}

Vector<uint8_t> quotedPrintableDecode(const Vector<uint8_t>& input)
{
    return quotedPrintableDecode(input.span());
}

Vector<uint8_t> quotedPrintableDecode(std::span<const uint8_t> data)
{
    Vector<uint8_t> out;
    for (size_t i = 0; i < data.size(); ++i) {
        char currentCharacter = data[i];
        if (currentCharacter != '=') {
            out.append(currentCharacter);
            continue;
        }
        // We are dealing with a '=xx' sequence.
        if (data.size() - i < 3) {
            // Unfinished = sequence, append as is.
            out.append(currentCharacter);
            continue;
        }
        char upperCharacter = data[++i];
        char lowerCharacter = data[++i];
        if (upperCharacter == '\r' && lowerCharacter == '\n')
            continue;

        if (!isASCIIHexDigit(upperCharacter) || !isASCIIHexDigit(lowerCharacter)) {
            // Invalid sequence, = followed by non hex digits, just insert the characters as is.
            out.append('=');
            out.append(upperCharacter);
            out.append(lowerCharacter);
            continue;
        }
        out.append(toASCIIHexValue(upperCharacter, lowerCharacter));
    }
    return out;
}

}
