/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 4, 2022.
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

#include <wtf/text/EscapedFormsForJSON.h>
#include <wtf/text/ParsingUtilities.h>
#include <wtf/text/StringBuilderInternals.h>
#include <wtf/text/WTFString.h>

namespace WTF {

template<typename OutputCharacterType, typename InputCharacterType>
ALWAYS_INLINE static void appendEscapedJSONStringContent(std::span<OutputCharacterType>& output, std::span<const InputCharacterType> input)
{
    for (; !input.empty(); skip(input, 1)) {
        auto character = input.front();
        if (LIKELY(character <= 0xFF)) {
            auto escaped = escapedFormsForJSON[character];
            if (LIKELY(!escaped)) {
                consume(output) = character;
                continue;
            }

            output[0] = '\\';
            output[1] = escaped;
            skip(output, 2);
            if (UNLIKELY(escaped == 'u')) {
                output[0] = '0';
                output[1] = '0';
                output[2] = upperNibbleToLowercaseASCIIHexDigit(character);
                output[3] = lowerNibbleToLowercaseASCIIHexDigit(character);
                skip(output, 4);
            }
            continue;
        }

        if (LIKELY(!U16_IS_SURROGATE(character))) {
            consume(output) = character;
            continue;
        }

        if (input.size() > 1) {
            auto next = input[1];
            bool isValidSurrogatePair = U16_IS_SURROGATE_LEAD(character) && U16_IS_TRAIL(next);
            if (isValidSurrogatePair) {
                output[0] = character;
                output[1] = next;
                skip(output, 2);
                skip(input, 1);
                continue;
            }
        }

        uint8_t upper = static_cast<uint32_t>(character) >> 8;
        uint8_t lower = static_cast<uint8_t>(character);
        output[0] = '\\';
        output[1] = 'u';
        output[2] = upperNibbleToLowercaseASCIIHexDigit(upper);
        output[3] = lowerNibbleToLowercaseASCIIHexDigit(upper);
        output[4] = upperNibbleToLowercaseASCIIHexDigit(lower);
        output[5] = lowerNibbleToLowercaseASCIIHexDigit(lower);
        skip(output, 6);
    }
}

} // namespace WTF
