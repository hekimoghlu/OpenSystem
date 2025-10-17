/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 14, 2023.
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
#include <wtf/StdLibExtras.h>

namespace JSC {

#if USE(JSVALUE64)
// According to C++ rules, a type used for the return signature of function with C linkage (i.e.
// 'extern "C"') needs to be POD; hence putting any constructors into it could cause either compiler
// warnings, or worse, a change in the ABI used to return these types.
struct UGPRPair {
    UCPURegister first;
    UCPURegister second;
};
static_assert(sizeof(UGPRPair) == sizeof(UCPURegister) * 2, "UGPRPair should fit in two machine registers");

constexpr UGPRPair makeUGPRPair(UCPURegister first, UCPURegister second) { return { first, second }; }

inline UGPRPair encodeResult(const void* a, const void* b)
{
    return makeUGPRPair(reinterpret_cast<UCPURegister>(a), reinterpret_cast<UCPURegister>(b));
}

inline void decodeResult(UGPRPair result, const void*& a, const void*& b)
{
    a = reinterpret_cast<void*>(result.first);
    b = reinterpret_cast<void*>(result.second);
}

inline void decodeResult(UGPRPair result, size_t& a, size_t& b)
{
    a = static_cast<size_t>(result.first);
    b = static_cast<size_t>(result.second);
}

#else // USE(JSVALUE32_64)
using UGPRPair = uint64_t;

#if CPU(BIG_ENDIAN)
constexpr UGPRPair makeUGPRPair(UCPURegister first, UCPURegister second) { return static_cast<uint64_t>(first) << 32 | second; }
#else
constexpr UGPRPair makeUGPRPair(UCPURegister first, UCPURegister second) { return static_cast<uint64_t>(second) << 32 | first; }
#endif

typedef union {
    struct {
        const void* a;
        const void* b;
    } pair;
    uint64_t i;
} UGPRPairEncoding;


inline UGPRPair encodeResult(const void* a, const void* b)
{
    return makeUGPRPair(reinterpret_cast<UCPURegister>(a), reinterpret_cast<UCPURegister>(b));
}

inline void decodeResult(UGPRPair result, const void*& a, const void*& b)
{
    UGPRPairEncoding u;
    u.i = result;
    a = u.pair.a;
    b = u.pair.b;
}

inline void decodeResult(UGPRPair result, size_t& a, size_t& b)
{
    UGPRPairEncoding u;
    u.i = result;
    a = std::bit_cast<size_t>(u.pair.a);
    b = std::bit_cast<size_t>(u.pair.b);
}

#endif // USE(JSVALUE32_64)

} // namespace JSC
