/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 30, 2025.
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

#include <span>
#include <string>

namespace WTF {

// Returns zero if arrays are equal, and non-zero otherwise. Execution time does not depend on array contents.
#if HAVE(TIMINGSAFE_BCMP)
inline int constantTimeMemcmp(std::span<const uint8_t> a, std::span<const uint8_t> b)
{
    RELEASE_ASSERT(a.size() == b.size());
    return timingsafe_bcmp(a.data(), b.data(), b.size());
}
#else
WTF_EXPORT_PRIVATE int constantTimeMemcmp(std::span<const uint8_t>, std::span<const uint8_t>);
#endif

}

using WTF::constantTimeMemcmp;
