/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 25, 2024.
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

#include <wtf/StdLibExtras.h>
#include <wtf/text/ASCIIFastPath.h>

namespace PAL {

template<size_t size> struct UCharByteFiller;
template<> struct UCharByteFiller<4> {
    static void copy(std::span<LChar> destination, std::span<const uint8_t> source)
    {
        memcpySpan(destination, source.first(4));
    }
    
    static void copy(std::span<UChar> destination, std::span<const uint8_t> source)
    {
        destination[0] = source[0];
        destination[1] = source[1];
        destination[2] = source[2];
        destination[3] = source[3];
    }
};
template<> struct UCharByteFiller<8> {
    static void copy(std::span<LChar> destination, std::span<const uint8_t> source)
    {
        memcpySpan(destination, source.first(8));
    }

    static void copy(std::span<UChar> destination, std::span<const uint8_t> source)
    {
        destination[0] = source[0];
        destination[1] = source[1];
        destination[2] = source[2];
        destination[3] = source[3];
        destination[4] = source[4];
        destination[5] = source[5];
        destination[6] = source[6];
        destination[7] = source[7];
    }
};

inline void copyASCIIMachineWord(std::span<LChar> destination, std::span<const uint8_t> source)
{
    UCharByteFiller<sizeof(WTF::MachineWord)>::copy(destination, source);
}

inline void copyASCIIMachineWord(std::span<UChar> destination, std::span<const uint8_t> source)
{
    UCharByteFiller<sizeof(WTF::MachineWord)>::copy(destination, source);
}

} // namespace PAL
