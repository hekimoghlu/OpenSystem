/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 7, 2022.
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

#include "CPU.h"

#include <wtf/PrintStream.h>

namespace JSC {

enum class Width : uint8_t {
    Width8,
    Width16,
    Width32,
    Width64,
    Width128,
};
static constexpr Width Width8 = Width::Width8;
static constexpr Width Width16 = Width::Width16;
static constexpr Width Width32 = Width::Width32;
static constexpr Width Width64 = Width::Width64;
static constexpr Width Width128 = Width::Width128;

enum class PreservedWidth : uint8_t {
    PreservesNothing = 0,
    Preserves64 = 1,
};
static constexpr PreservedWidth PreservesNothing = PreservedWidth::PreservesNothing;
static constexpr PreservedWidth Preserves64 = PreservedWidth::Preserves64;

ALWAYS_INLINE constexpr Width widthForBytes(unsigned bytes)
{
    switch (bytes) {
    case 0:
    case 1:
        return Width8;
    case 2:
        return Width16;
    case 3:
    case 4:
        return Width32;
    case 5:
    case 6:
    case 7:
    case 8:
        return Width64;
    default:
        return Width128;
    }
}

ALWAYS_INLINE constexpr unsigned bytesForWidth(Width width)
{
    switch (width) {
    case Width8:
        return 1;
    case Width16:
        return 2;
    case Width32:
        return 4;
    case Width64:
        return 8;
    case Width128:
        return 16;
    }
    RELEASE_ASSERT_NOT_REACHED();
    return 0;
}

ALWAYS_INLINE constexpr unsigned alignmentForWidth(Width width)
{
    switch (width) {
    case Width8:
        return 1;
    case Width16:
        return 2;
    case Width32:
        return 4;
    case Width64:
        return 8;
    case Width128:
        return 8;
    }
    RELEASE_ASSERT_NOT_REACHED();
    return 0;
}

inline constexpr uint64_t mask(Width width)
{
    switch (width) {
    case Width8:
        return 0x00000000000000ffllu;
    case Width16:
        return 0x000000000000ffffllu;
    case Width32:
        return 0x00000000ffffffffllu;
    case Width64:
        return 0xffffffffffffffffllu;
    case Width128:
        RELEASE_ASSERT_NOT_REACHED();
        return 0;
    }
    RELEASE_ASSERT_NOT_REACHED();
    return 0;
}

constexpr Width pointerWidth()
{
    if (isAddress64Bit())
        return Width64;
    return Width32;
}

constexpr Width registerWidth()
{
    if (isRegister64Bit())
        return Width64;
    return Width32;
}

inline Width canonicalWidth(Width width)
{
    return std::max(Width32, width);
}

inline bool isCanonicalWidth(Width width)
{
    return width >= Width32;
}

} // namespace JSC

namespace WTF {

void printInternal(PrintStream& out, JSC::Width width);

} // namespace WTF
