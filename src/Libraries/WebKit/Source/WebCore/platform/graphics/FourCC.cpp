/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 23, 2021.
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
#include "FourCC.h"

namespace WebCore {

std::optional<FourCC> FourCC::fromString(StringView string)
{
    if (string.length() != 4 || !string.containsOnlyASCII())
        return std::nullopt;

    std::array<char, 5> code {
        static_cast<char>(string[0]),
        static_cast<char>(string[1]),
        static_cast<char>(string[2]),
        static_cast<char>(string[3]),
        '\0'
    };
    return FourCC { std::span { code } };
}

}
