/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 28, 2024.
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
#include "FileSizeFormatter.h"

#include <wtf/text/MakeString.h>

namespace PAL {

#if !PLATFORM(COCOA)

AtomString fileSizeDescription(uint64_t size)
{
    // FIXME: These strings should be localized, but that would require bringing LocalizedStrings into PAL.
    // See <https://bugs.webkit.org/show_bug.cgi?id=179019> for more details.
    if (size < 1000)
        return makeAtomString(size, " bytes"_s);
    if (size < 1000000)
        return makeAtomString(FormattedNumber::fixedWidth(size / 1000., 1), " KB"_s);
    if (size < 1000000000)
        return makeAtomString(FormattedNumber::fixedWidth(size / 1000000., 1), " MB"_s);
    return makeAtomString(FormattedNumber::fixedWidth(size / 1000000000., 1), " GB"_s);
}

#endif

} // namespace PAL
