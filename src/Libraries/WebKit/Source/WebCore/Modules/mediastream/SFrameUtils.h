/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 1, 2022.
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

#if ENABLE(WEB_RTC)

#include <wtf/Vector.h>

namespace WebCore {

using SFrameCompatibilityPrefixBuffer = std::variant<std::span<const uint8_t>, Vector<uint8_t>>;

size_t computeH264PrefixOffset(std::span<const uint8_t>);
SFrameCompatibilityPrefixBuffer computeH264PrefixBuffer(std::span<const uint8_t>);

WEBCORE_EXPORT bool needsRbspUnescaping(std::span<const uint8_t>);
WEBCORE_EXPORT Vector<uint8_t> fromRbsp(std::span<const uint8_t>);
WEBCORE_EXPORT void toRbsp(Vector<uint8_t>&, size_t);

size_t computeVP8PrefixOffset(std::span<const uint8_t>);
SFrameCompatibilityPrefixBuffer computeVP8PrefixBuffer(std::span<const uint8_t>);

static inline Vector<uint8_t, 8> encodeBigEndian(uint64_t value)
{
    Vector<uint8_t, 8> result(8);
    for (int i = 7; i >= 0; --i) {
        result[i] = value & 0xff;
        value = value >> 8;
    }
    return result;
}

} // namespace WebCore

#endif // ENABLE(WEB_RTC)
