/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 30, 2024.
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
#ifndef mach_o_Misc_h
#define mach_o_Misc_h

#include <stdint.h>

#include <span>
#include <string_view>

#include "MachODefines.h"
#include "Error.h"

namespace mach_o {

struct Header;

/// Returns true if (addLHS + addRHS) > b, or if the add overflowed
template<typename T>
VIS_HIDDEN inline bool greaterThanAddOrOverflow(uint32_t addLHS, uint32_t addRHS, T b) {
    uint32_t sum;
    if (__builtin_add_overflow(addLHS, addRHS, &sum) )
        return true;
    return (sum > b);
}

/// Returns true if (addLHS + addRHS) > b, or if the add overflowed
template<typename T>
VIS_HIDDEN inline bool greaterThanAddOrOverflow(uint64_t addLHS, uint64_t addRHS, T b) {
    uint64_t sum;
    if (__builtin_add_overflow(addLHS, addRHS, &sum) )
        return true;
    return (sum > b);
}


uint64_t    read_uleb128(const uint8_t*& p, const uint8_t* end, bool& malformed) VIS_HIDDEN;
uint64_t    read_uleb128(std::span<const uint8_t>& buffer, bool& malformed) VIS_HIDDEN;
int64_t     read_sleb128(const uint8_t*& p, const uint8_t* end, bool& malformed) VIS_HIDDEN;
uint32_t	uleb128_size(uint64_t value) VIS_HIDDEN;

inline void pageAlign4K(uint64_t& value)
{
    value = ((value + 0xFFF) & (-0x1000));
}

inline void pageAlign16K(uint64_t& value)
{
    value = ((value + 0x3FFF) & (-0x4000));
}

// used to walk fat/thin files and get all mach-o headers
Error forEachHeader(std::span<const uint8_t> buffer, std::string_view path,
                    void (^callback)(const Header* sliceHeader, size_t sliceLength, bool& stop)) VIS_HIDDEN;

} // namespace mach_o

#endif /* mach_o_Misc_h */
