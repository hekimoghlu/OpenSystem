/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 26, 2025.
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
#ifndef BitUtils_h
#define BitUtils_h

// TODO: This can be removed once <bit> works completely in Driverkit builds

#include <bit>
#include <cstdint>

namespace lsl {
inline
__attribute__ ((__const__))
uint64_t bit_ceil(uint64_t value) {
    value--;
    value |= value >> 1;
    value |= value >> 2;
    value |= value >> 4;
    value |= value >> 8;
    value |= value >> 16;
    value |= value >> 32;
    value++;
    return value;
}

inline
__attribute__ ((__const__))
uint64_t bit_width(uint64_t value) {
    return std::numeric_limits<uint64_t>::digits - std::countl_zero(value);
}

template<uint64_t SIZE>
inline
__attribute__((__const__))
uint64_t roundDownToAligned(uint64_t value) {
    static_assert(std::popcount(SIZE) == 1);
    return (value & ~(SIZE-1));
}

template<uint64_t SIZE>
inline
__attribute__((__const__))
uint64_t roundToNextAligned(uint64_t value) {
    static_assert(std::popcount(SIZE) == 1);
    return (value+(SIZE-1) & (-1*SIZE));
}

inline
uint64_t roundDownToAligned(uint64_t alignment, uint64_t value) {
    return (value & ~(alignment-1));
}

inline
uint64_t roundToNextAligned(uint64_t alignment, uint64_t value) {
    return (value+(alignment-1) & (-1*alignment));
}

};

#endif /* BitUtils_h */
