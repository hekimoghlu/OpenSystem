/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 23, 2023.
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
#include "YarrFlags.h"

#include <wtf/OptionSet.h>
#include <wtf/text/StringView.h>

namespace JSC { namespace Yarr {

std::optional<OptionSet<Flags>> parseFlags(StringView string)
{
    OptionSet<Flags> flags;
    for (auto character : string.codeUnits()) {
        switch (character) {
#define JSC_HANDLE_REGEXP_FLAG(key, name, lowerCaseName, _) \
        case key: \
            if (flags.contains(Flags::name)) \
                return std::nullopt; \
            flags.add(Flags::name); \
            break;

        JSC_REGEXP_FLAGS(JSC_HANDLE_REGEXP_FLAG)

#undef JSC_HANDLE_REGEXP_FLAG

        default:
            return std::nullopt;
        }
    }

    // Can only specify one of 'u' and 'v' flags.
    if (flags.contains(Flags::Unicode) && flags.contains(Flags::UnicodeSets))
        return std::nullopt;

    return std::make_optional(flags);
}

FlagsString flagsString(OptionSet<Flags> flags)
{
    FlagsString string;
    unsigned index = 0;

#define JSC_WRITE_REGEXP_FLAG(key, name, lowerCaseName, _) \
    do { \
        if (flags.contains(Flags::name)) \
            string[index++] = key; \
    } while (0);

    JSC_REGEXP_FLAGS(JSC_WRITE_REGEXP_FLAG)

#undef JSC_WRITE_REGEXP_FLAG

    ASSERT(index < string.size());
    string[index] = 0;
    return string;
}

} } // namespace JSC::Yarr
