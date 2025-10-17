/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 18, 2022.
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
#include "TextCodec.h"
#include <unicode/uchar.h>
#include <wtf/TZoneMallocInlines.h>
#include <wtf/text/WTFString.h>
#include <wtf/unicode/CharacterNames.h>

#include <array>
#include <cstdio>

namespace PAL {

WTF_MAKE_TZONE_ALLOCATED_IMPL(TextCodec);

std::span<char> TextCodec::getUnencodableReplacement(char32_t codePoint, UnencodableHandling handling, UnencodableReplacementArray& replacement)
{
    ASSERT(!(codePoint > UCHAR_MAX_VALUE));

    // The Encoding Standard doesn't have surrogate code points in the input, but that would require
    // scanning and potentially manipulating inputs ahead of time. Instead handle them at the last
    // possible point.
    if (U_IS_SURROGATE(codePoint))
        codePoint = replacementCharacter;

    switch (handling) {
    case UnencodableHandling::Entities: {
        int count = SAFE_SPRINTF(std::span { replacement }, "&#%u;", static_cast<unsigned>(codePoint));
        ASSERT(count >= 0);
        return std::span { replacement }.first(std::max<int>(0, count));
    }
    case UnencodableHandling::URLEncodedEntities: {
        int count = SAFE_SPRINTF(std::span { replacement }, "%%26%%23%u%%3B", static_cast<unsigned>(codePoint));
        ASSERT(count >= 0);
        return std::span { replacement }.first(std::max<int>(0, count));
    } }

    ASSERT_NOT_REACHED();
    replacement[0] = '\0';
    return std::span { replacement }.first(0);
}

} // namespace PAL
