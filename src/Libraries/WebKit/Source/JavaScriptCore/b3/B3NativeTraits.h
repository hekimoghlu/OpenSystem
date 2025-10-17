/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 11, 2023.
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

#if ENABLE(B3_JIT)

#include "B3Bank.h"
#include "B3Type.h"
#include "B3Width.h"

namespace JSC { namespace B3 {

template<typename> struct NativeTraits;

template<> struct NativeTraits<int8_t> {
    typedef int32_t CanonicalType;
    static constexpr Bank bank = GP;
    static constexpr Width width = Width8;
    static constexpr Type type = Int32;
};

template<> struct NativeTraits<uint8_t> {
    typedef int32_t CanonicalType;
    static constexpr Bank bank = GP;
    static constexpr Width width = Width8;
    static constexpr Type type = Int32;
};

template<> struct NativeTraits<int16_t> {
    typedef int32_t CanonicalType;
    static constexpr Bank bank = GP;
    static constexpr Width width = Width16;
    static constexpr Type type = Int32;
};

template<> struct NativeTraits<uint16_t> {
    typedef int32_t CanonicalType;
    static constexpr Bank bank = GP;
    static constexpr Width width = Width16;
    static constexpr Type type = Int32;
};

template<> struct NativeTraits<int32_t> {
    typedef int32_t CanonicalType;
    static constexpr Bank bank = GP;
    static constexpr Width width = Width32;
    static constexpr Type type = Int32;
};

template<> struct NativeTraits<uint32_t> {
    typedef int32_t CanonicalType;
    static constexpr Bank bank = GP;
    static constexpr Width width = Width32;
    static constexpr Type type = Int32;
};

template<> struct NativeTraits<int64_t> {
    typedef int64_t CanonicalType;
    static constexpr Bank bank = GP;
    static constexpr Width width = Width64;
    static constexpr Type type = Int64;
};

template<> struct NativeTraits<uint64_t> {
    typedef int64_t CanonicalType;
    static constexpr Bank bank = GP;
    static constexpr Width width = Width64;
    static constexpr Type type = Int64;
};

template<> struct NativeTraits<float> {
    typedef float CanonicalType;
    static constexpr Bank bank = FP;
    static constexpr Width width = Width32;
    static constexpr Type type = Float;
};

template<> struct NativeTraits<double> {
    typedef double CanonicalType;
    static constexpr Bank bank = FP;
    static constexpr Width width = Width64;
    static constexpr Type type = Double;
};

} } // namespace JSC::B3

#endif // ENABLE(B3_JIT)

