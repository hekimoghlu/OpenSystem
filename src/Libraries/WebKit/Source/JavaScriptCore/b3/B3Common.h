/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 2, 2024.
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

#include "CPU.h"
#include "GPRInfo.h"
#include "JSExportMacros.h"
#include "Options.h"

namespace JSC { namespace B3 {

class Procedure;

extern const char* const tierName;

enum B3CompilationMode {
    B3Mode,
    AirMode
};

JS_EXPORT_PRIVATE bool shouldDumpIR(Procedure&, B3CompilationMode);
bool shouldDumpIRAtEachPhase(B3CompilationMode);
bool shouldValidateIR();
bool shouldValidateIRAtEachPhase();
bool shouldSaveIRBeforePhase();

template<typename IntType>
static IntType chillDiv(IntType numerator, IntType denominator)
{
    if (!denominator)
        return 0;
    if (denominator == -1 && numerator == std::numeric_limits<IntType>::min())
        return std::numeric_limits<IntType>::min();
    return numerator / denominator;
}

template<typename IntType>
static IntType chillMod(IntType numerator, IntType denominator)
{
    if (!denominator)
        return 0;
    if (denominator == -1 && numerator == std::numeric_limits<IntType>::min())
        return 0;
    return numerator % denominator;
}

template<typename IntType>
static IntType chillUDiv(IntType numerator, IntType denominator)
{
    typedef typename std::make_unsigned<IntType>::type UnsignedIntType;
    UnsignedIntType unsignedNumerator = static_cast<UnsignedIntType>(numerator);
    UnsignedIntType unsignedDenominator = static_cast<UnsignedIntType>(denominator);
    if (!unsignedDenominator)
        return 0;
    return unsignedNumerator / unsignedDenominator;
}

template<typename IntType>
static IntType chillUMod(IntType numerator, IntType denominator)
{
    typedef typename std::make_unsigned<IntType>::type UnsignedIntType;
    UnsignedIntType unsignedNumerator = static_cast<UnsignedIntType>(numerator);
    UnsignedIntType unsignedDenominator = static_cast<UnsignedIntType>(denominator);
    if (!unsignedDenominator)
        return 0;
    return unsignedNumerator % unsignedDenominator;
}

template<typename IntType>
static IntType rotateRight(IntType value, int32_t shift)
{
    typedef typename std::make_unsigned<IntType>::type UnsignedIntType;
    UnsignedIntType uValue = static_cast<UnsignedIntType>(value);
    int32_t bits = sizeof(IntType) * 8;
    int32_t mask = bits - 1;
    shift &= mask;
    return (uValue >> shift) | (uValue << ((bits - shift) & mask));
}

template<typename IntType>
static IntType rotateLeft(IntType value, int32_t shift)
{
    typedef typename std::make_unsigned<IntType>::type UnsignedIntType;
    UnsignedIntType uValue = static_cast<UnsignedIntType>(value);
    int32_t bits = sizeof(IntType) * 8;
    int32_t mask = bits - 1;
    shift &= mask;
    return (uValue << shift) | (uValue >> ((bits - shift) & mask));
}

inline unsigned defaultOptLevel()
{
    // This should almost always return 2, but we allow this default to be lowered for testing. Some
    // components will deliberately set the optLevel.
    return Options::defaultB3OptLevel();
}

GPRReg extendedOffsetAddrRegister();

} } // namespace JSC::B3

#endif // ENABLE(B3_JIT)
