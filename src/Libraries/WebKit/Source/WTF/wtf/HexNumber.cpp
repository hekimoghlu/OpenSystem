/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 15, 2024.
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
#include "HexNumber.h"

#include <wtf/ASCIICType.h>
#include <wtf/CheckedArithmetic.h>
#include <wtf/IndexedRange.h>
#include <wtf/PrintStream.h>
#include <wtf/StdLibExtras.h>
#include <wtf/text/StringView.h>

namespace WTF {

namespace Internal {

std::span<LChar> appendHex(std::span<LChar> buffer, std::uintmax_t number, unsigned minimumDigits, HexConversionMode mode)
{
    size_t startIndex = buffer.size();
    auto& hexDigits = hexDigitsForMode(mode);
    do {
        buffer[--startIndex] = hexDigits[number & 0xF];
        number >>= 4;
    } while (number);
    auto startIndexWithLeadingZeros = buffer.size() - std::min<size_t>(minimumDigits, buffer.size());
    if (startIndex > startIndexWithLeadingZeros) {
        memsetSpan(buffer.subspan(startIndexWithLeadingZeros, startIndex - startIndexWithLeadingZeros), '0');
        startIndex = startIndexWithLeadingZeros;
    }
    return buffer.subspan(startIndex);
}

}

void printInternal(PrintStream& out, HexNumberBuffer buffer)
{
    out.print(StringView(buffer.span()));
}

static void toHexInternal(std::span<const uint8_t> values, std::span<LChar> hexadecimalOutput)
{
    for (auto [i, digestValue] : indexedRange(values)) {
        hexadecimalOutput[i * 2] = upperNibbleToASCIIHexDigit(digestValue);
        hexadecimalOutput[i * 2 + 1] = lowerNibbleToASCIIHexDigit(digestValue);
    }
}

CString toHexCString(std::span<const uint8_t> values)
{
    std::span<char> buffer;
    auto result = CString::newUninitialized(CheckedSize(values.size()) * 2U, buffer);
    toHexInternal(values, byteCast<LChar>(buffer));
    return result;
}

String toHexString(std::span<const uint8_t> values)
{
    std::span<LChar> buffer;
    auto result = String::createUninitialized(CheckedSize(values.size()) * 2U, buffer);
    toHexInternal(values, buffer);
    return result;
}

} // namespace WTF
