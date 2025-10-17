/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 25, 2025.
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

#include <algorithm>
#include <limits.h>
#include <wtf/Compiler.h>

// This file contains a bunch of helper functions for decoding LEB numbers.
// See https://en.wikipedia.org/wiki/LEB128 for more information about the
// LEB format.

namespace WTF { namespace LEBDecoder {

template<typename T>
constexpr size_t maxByteLength()
{
    constexpr size_t numBits = sizeof(T) * CHAR_BIT;
    return (numBits - 1) / 7 + 1; // numBits / 7 rounding up.
}

template<typename T>
constexpr unsigned lastByteMask()
{
    constexpr size_t numBits = sizeof(T) * CHAR_BIT;
    static_assert(numBits % 7);
    return ~((1U << (numBits % 7)) - 1);
}

template<typename T>
inline bool WARN_UNUSED_RETURN decodeUInt(std::span<const uint8_t> bytes, size_t& offset, T& result)
{
    static_assert(std::is_unsigned_v<T>);
    if (bytes.size() <= offset)
        return false;
    result = 0;
    unsigned shift = 0;
    size_t last = std::min(maxByteLength<T>(), bytes.size() - offset) - 1;
    for (unsigned i = 0; true; ++i) {
        uint8_t byte = bytes[offset++];
        result |= static_cast<T>(byte & 0x7f) << shift;
        shift += 7;
        if (!(byte & 0x80))
            return !(((maxByteLength<T>() - 1) == i && (byte & lastByteMask<T>())));
        if (i == last)
            return false;
    }
    RELEASE_ASSERT_NOT_REACHED();
    return true;
}

template<typename T>
inline bool WARN_UNUSED_RETURN decodeInt(std::span<const uint8_t> bytes, size_t& offset, T& result)
{
    static_assert(std::is_signed_v<T>);
    if (bytes.size() <= offset)
        return false;
    using UnsignedT = typename std::make_unsigned<T>::type;
    result = 0;
    unsigned shift = 0;
    size_t last = std::min(maxByteLength<T>(), bytes.size() - offset) - 1;
    uint8_t byte;
    for (unsigned i = 0; true; ++i) {
        byte = bytes[offset++];
        result |= static_cast<T>(static_cast<UnsignedT>(byte & 0x7f) << shift);
        shift += 7;
        if (!(byte & 0x80)) {
            if ((maxByteLength<T>() - 1) == i) {
                if (!(byte & 0x40)) {
                    // This is a non-sign-extended, positive number. Then, the remaining bits should be (lastByteMask<T>() >> 1).
                    // For example, in the int32_t case, the last byte should be less than 0b00000111, since 7 * 4 + 3 = 31.
                    if (byte & (lastByteMask<T>() >> 1))
                        return false;
                } else {
                    // This is sign-extended, negative number. Then, zero should not exists in (lastByteMask<T>() >> 1) bits except for the top bit.
                    // For example, in the int32_t case, the last byte should be 0b01111XXX and 1 part must be 1. Since we already checked 0x40 is 1,
                    // middle [3,5] bits must be zero (e.g. 0b01000111 is invalid). We convert 0b01111XXX =(| 0x80)=> 0b11111XXX =(~)=> 0b00000YYY.
                    // And check that we do not have 1 in upper 5 bits.
                    if (static_cast<uint8_t>(~(byte | 0x80)) & (lastByteMask<T>() >> 1))
                        return false;
                }
            }
            break;
        }
        if (i == last)
            return false;
    }

    const size_t numBits = sizeof(T) * CHAR_BIT;
    if (shift < numBits && (byte & 0x40))
        result = static_cast<T>(static_cast<UnsignedT>(result) | (static_cast<UnsignedT>(-1) << shift));
    return true;
}

inline bool WARN_UNUSED_RETURN decodeUInt32(std::span<const uint8_t> bytes, size_t& offset, uint32_t& result)
{
    return decodeUInt<uint32_t>(bytes, offset, result);
}

inline bool WARN_UNUSED_RETURN decodeUInt64(std::span<const uint8_t> bytes, size_t& offset, uint64_t& result)
{
    return decodeUInt<uint64_t>(bytes, offset, result);
}

inline bool WARN_UNUSED_RETURN decodeInt32(std::span<const uint8_t> bytes, size_t& offset, int32_t& result)
{
    return decodeInt<int32_t>(bytes, offset, result);
}

inline bool WARN_UNUSED_RETURN decodeInt64(std::span<const uint8_t> bytes, size_t& offset, int64_t& result)
{
    return decodeInt<int64_t>(bytes, offset, result);
}

} } // WTF::LEBDecoder
