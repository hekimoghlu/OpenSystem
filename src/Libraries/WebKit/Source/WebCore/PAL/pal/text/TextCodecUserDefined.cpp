/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 30, 2022.
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
#include "TextCodecUserDefined.h"

#include <array>
#include <wtf/TZoneMallocInlines.h>
#include <wtf/text/CString.h>
#include <wtf/text/StringBuilder.h>
#include <wtf/text/WTFString.h>

namespace PAL {

WTF_MAKE_TZONE_ALLOCATED_IMPL(TextCodecUserDefined);

void TextCodecUserDefined::registerEncodingNames(EncodingNameRegistrar registrar)
{
    registrar("x-user-defined"_s, "x-user-defined"_s);
}

void TextCodecUserDefined::registerCodecs(TextCodecRegistrar registrar)
{
    registrar("x-user-defined"_s, [] {
        return makeUnique<TextCodecUserDefined>();
    });
}

String TextCodecUserDefined::decode(std::span<const uint8_t> bytes, bool, bool, bool&)
{
    StringBuilder result;
    result.reserveCapacity(bytes.size());
    for (char byte : bytes)
        result.append(static_cast<UChar>(byte & 0xF7FF));
    return result.toString();
}

static Vector<uint8_t> encodeComplexUserDefined(StringView string, UnencodableHandling handling)
{
    Vector<uint8_t> result;

    for (auto character : string.codePoints()) {
        int8_t signedByte = character;
        if ((signedByte & 0xF7FF) == character)
            result.append(signedByte);
        else {
            // No way to encode this character with x-user-defined.
            UnencodableReplacementArray replacement;
            result.append(TextCodec::getUnencodableReplacement(character, handling, replacement));
        }
    }

    return result;
}

Vector<uint8_t> TextCodecUserDefined::encode(StringView string, UnencodableHandling handling) const
{
    {
        Vector<uint8_t> result(string.length());
        size_t index = 0;

        // Convert and simultaneously do a check to see if it's all ASCII.
        UChar ored = 0;
        for (auto character : string.codeUnits()) {
            result[index++] = character;
            ored |= character;
        }

        if (!(ored & 0xFF80))
            return result;
    }

    // If it wasn't all ASCII, call the function that handles more-complex cases.
    return encodeComplexUserDefined(string, handling);
}

} // namespace PAL
