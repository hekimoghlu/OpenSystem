/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 22, 2023.
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
#include "TextCodecReplacement.h"

#include <wtf/Function.h>
#include <wtf/TZoneMallocInlines.h>
#include <wtf/text/WTFString.h>
#include <wtf/unicode/CharacterNames.h>

namespace PAL {

WTF_MAKE_TZONE_ALLOCATED_IMPL(TextCodecReplacement);

void TextCodecReplacement::registerEncodingNames(EncodingNameRegistrar registrar)
{
    registrar("replacement"_s, "replacement"_s);

    registrar("csiso2022kr"_s, "replacement"_s);
    registrar("hz-gb-2312"_s, "replacement"_s);
    registrar("iso-2022-cn"_s, "replacement"_s);
    registrar("iso-2022-cn-ext"_s, "replacement"_s);
    registrar("iso-2022-kr"_s, "replacement"_s);
}

void TextCodecReplacement::registerCodecs(TextCodecRegistrar registrar)
{
    registrar("replacement"_s, [] {
        return makeUnique<TextCodecReplacement>();
    });
}

String TextCodecReplacement::decode(std::span<const uint8_t>, bool, bool, bool& sawError)
{
    sawError = true;
    if (m_sentEOF)
        return emptyString();
    m_sentEOF = true;
    return span(replacementCharacter);
}

Vector<uint8_t> TextCodecReplacement::encode(StringView string, UnencodableHandling) const
{
    return TextCodecUTF8::encodeUTF8(string);
}

} // namespace PAL
