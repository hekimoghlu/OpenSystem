/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 17, 2023.
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

#include <wtf/text/LChar.h>

namespace WTF {
namespace Unicode {

enum class ConversionResultCode : uint8_t {
    Success, // conversion successful
    SourceInvalid, // source sequence is invalid/malformed
    TargetExhausted // insufficient room in target for conversion
};

template<typename CharacterType> struct ConversionResult {
    ConversionResultCode code { };
    std::span<CharacterType> buffer { };
    bool isAllASCII { };
};

WTF_EXPORT_PRIVATE ConversionResult<char16_t> convert(std::span<const char8_t>, std::span<char16_t>);
WTF_EXPORT_PRIVATE ConversionResult<char8_t> convert(std::span<const char16_t>, std::span<char8_t>);
WTF_EXPORT_PRIVATE ConversionResult<char8_t> convert(std::span<const LChar>, std::span<char8_t>);

// Invalid sequences are converted to the replacement character.
WTF_EXPORT_PRIVATE ConversionResult<char16_t> convertReplacingInvalidSequences(std::span<const char8_t>, std::span<char16_t>);
WTF_EXPORT_PRIVATE ConversionResult<char8_t> convertReplacingInvalidSequences(std::span<const char16_t>, std::span<char8_t>);

WTF_EXPORT_PRIVATE bool equal(std::span<const char16_t>, std::span<const char8_t>);
WTF_EXPORT_PRIVATE bool equal(std::span<const LChar>, std::span<const char8_t>);

// The checkUTF8 function checks the UTF-8 and returns a span of the valid complete
// UTF-8 sequences at the start of the passed-in characters, along with the UTF-16
// length and an indication if all were ASCII.
struct CheckedUTF8 {
    std::span<const char8_t> characters { };
    size_t lengthUTF16 { };
    bool isAllASCII { };
};
WTF_EXPORT_PRIVATE CheckedUTF8 checkUTF8(std::span<const char8_t>);

// The computeUTF16LengthWithHash function returns a length and hash of 0 if the
// source is exhausted or invalid. The hash is of the computeHashAndMaskTop8Bits variety.
struct UTF16LengthWithHash {
    size_t lengthUTF16 { };
    unsigned hash { };
};
WTF_EXPORT_PRIVATE UTF16LengthWithHash computeUTF16LengthWithHash(std::span<const char8_t>);

} // namespace Unicode
} // namespace WTF
