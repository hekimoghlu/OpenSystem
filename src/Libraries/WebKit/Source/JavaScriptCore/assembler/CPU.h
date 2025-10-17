/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 6, 2023.
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

#include "JSExportMacros.h"
#include <wtf/NumberOfCores.h>
#include <wtf/StdIntExtras.h>

namespace JSC {

using UCPUStrictInt32 = UCPURegister;

constexpr bool isARMv7IDIVSupported()
{
#if HAVE(ARM_IDIV_INSTRUCTIONS)
    return true;
#else
    return false;
#endif
}

constexpr bool isARM_THUMB2()
{
#if CPU(ARM_THUMB2)
    return true;
#else
    return false;
#endif
}

constexpr bool hasUnalignedFPMemoryAccess()
{
    return !isARM_THUMB2();
}

constexpr bool isARM64()
{
#if CPU(ARM64)
    return true;
#else
    return false;
#endif
}

constexpr bool isARM64E()
{
#if CPU(ARM64E)
    return true;
#else
    return false;
#endif
}

#if CPU(ARM64)
#if CPU(ARM64E)
JS_EXPORT_PRIVATE bool isARM64E_FPAC();
#else
constexpr bool isARM64E_FPAC() { return false; }
#endif

#if CPU(ARM64E) || OS(MAC_OS_X)
// ARM64E or all macOS ARM64 CPUs have LSE.
constexpr bool isARM64_LSE() { return true; }
#else
JS_EXPORT_PRIVATE bool isARM64_LSE();
#endif

#else // not CPU(ARM64)
constexpr bool isARM64_LSE() { return false; }
constexpr bool isARM64E_FPAC() { return false; }
#endif

constexpr bool isX86()
{
#if CPU(X86_64) || CPU(X86)
    return true;
#else
    return false;
#endif
}

constexpr bool isX86_64()
{
#if CPU(X86_64)
    return true;
#else
    return false;
#endif
}

#if CPU(X86_64)
JS_EXPORT_PRIVATE bool isX86_64_AVX();
#else
constexpr bool isX86_64_AVX()
{
    return false;
}
#endif

constexpr bool isRISCV64()
{
#if CPU(RISCV64)
    return true;
#else
    return false;
#endif
}

constexpr bool is64Bit()
{
#if USE(JSVALUE64)
    return true;
#else
    return false;
#endif
}

constexpr bool is32Bit()
{
    return !is64Bit();
}

constexpr bool isAddress64Bit()
{
    return sizeof(void*) == 8;
}

constexpr bool isAddress32Bit()
{
    return !isAddress64Bit();
}

constexpr size_t registerSize()
{
#if CPU(REGISTER64)
    return 8;
#elif CPU(REGISTER32)
    return 4;
#else
#  error "Unknown register size"
#endif
}

constexpr bool isRegister64Bit()
{
    return registerSize() == 8;
}

constexpr bool isRegister32Bit()
{
    return registerSize() == 4;
}

inline bool optimizeForARMv7IDIVSupported();
inline bool optimizeForARM64();
inline bool optimizeForX86();
inline bool optimizeForX86_64();
inline bool hasSensibleDoubleToInt();

#if PLATFORM(MAC) || PLATFORM(MACCATALYST)
bool isKernOpenSource();
#endif

#if (CPU(X86) || CPU(X86_64)) && OS(DARWIN)
bool isKernTCSMAvailable();
bool enableKernTCSM();
int kernTCSMAwareNumberOfProcessorCores();
int64_t hwL3CacheSize();
int32_t hwPhysicalCPUMax();
#else
ALWAYS_INLINE bool isKernTCSMAvailable() { return false; }
ALWAYS_INLINE bool enableKernTCSM() { return false; }
ALWAYS_INLINE int kernTCSMAwareNumberOfProcessorCores() { return WTF::numberOfProcessorCores(); }
ALWAYS_INLINE int64_t hwL3CacheSize() { return 0; }
ALWAYS_INLINE int32_t hwPhysicalCPUMax() { return kernTCSMAwareNumberOfProcessorCores(); }
#endif

constexpr size_t prologueStackPointerDelta()
{
#if ENABLE(C_LOOP)
    // Prologue saves the framePointerRegister and linkRegister
    return 2 * sizeof(CPURegister);
#elif CPU(X86_64)
    // Prologue only saves the framePointerRegister
    return sizeof(CPURegister);
#elif CPU(ARM_THUMB2) || CPU(ARM64) || CPU(RISCV64)
    // Prologue saves the framePointerRegister and linkRegister
    return 2 * sizeof(CPURegister);
#else
#error unsupported architectures
#endif
}



} // namespace JSC

