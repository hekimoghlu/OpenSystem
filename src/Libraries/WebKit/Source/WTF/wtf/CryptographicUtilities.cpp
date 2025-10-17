/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 21, 2024.
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
#include <wtf/CryptographicUtilities.h>
#include <wtf/ZippedRange.h>

namespace WTF {

#if !HAVE(TIMINGSAFE_BCMP)
NEVER_INLINE int constantTimeMemcmp(std::span<const uint8_t> a, std::span<const uint8_t> b)
{
    RELEASE_ASSERT(a.size() == b.size());

    uint8_t result = 0;
    for (auto [value1, value2] : zippedRange(a, b))
        result |= value1 ^ value2;
    return result;
}
#endif

}
