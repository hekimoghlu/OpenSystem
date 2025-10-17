/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 8, 2025.
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
#include <wtf/SixCharacterHash.h>

#include <wtf/ASCIICType.h>

namespace WTF {

unsigned sixCharacterHashStringToInteger(std::span<const char, 6> string)
{
    unsigned hash = 0;

    for (auto c : string) {
        hash *= 62;
        RELEASE_ASSERT(c); // FIXME: Why does this need to be a RELEASE_ASSERT?
        if (isASCIIUpper(c)) {
            hash += c - 'A';
            continue;
        }
        if (isASCIILower(c)) {
            hash += c - 'a' + 26;
            continue;
        }
        ASSERT(isASCIIDigit(c));
        hash += c - '0' + 26 * 2;
    }

    return hash;
}

std::array<char, 6> integerToSixCharacterHashString(unsigned hash)
{
    static constexpr std::array<char, 62> table {
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
        'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'
    };
    std::array<char, 6> buffer;
    unsigned accumulator = hash;
    for (unsigned i = 6; i--;) {
        buffer[i] = table[accumulator % 62];
        accumulator /= 62;
    }
    return buffer;
}

} // namespace WTF
