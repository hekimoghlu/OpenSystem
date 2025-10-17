/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 7, 2023.
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

#include "LLIntCommon.h"
#include "StructureID.h"
#include <wtf/Assertions.h>
#include <wtf/Gigacage.h>

#if ENABLE(C_LOOP)
#define OFFLINE_ASM_C_LOOP 1
#define OFFLINE_ASM_ARMv7 0
#define OFFLINE_ASM_ARM64 0
#define OFFLINE_ASM_ARM64E 0
#define OFFLINE_ASM_X86_64 0
#define OFFLINE_ASM_ARMv7k 0
#define OFFLINE_ASM_ARMv7s 0
#define OFFLINE_ASM_RISCV64 0

#else // ENABLE(C_LOOP)

#define OFFLINE_ASM_C_LOOP 0

#ifdef __ARM_ARCH_7K__
#define OFFLINE_ASM_ARMv7k 1
#else
#define OFFLINE_ASM_ARMv7k 0
#endif

#ifdef __ARM_ARCH_7S__
#define OFFLINE_ASM_ARMv7s 1
#else
#define OFFLINE_ASM_ARMv7s 0
#endif

#if CPU(ARM_THUMB2)
#define OFFLINE_ASM_ARMv7 1
#else
#define OFFLINE_ASM_ARMv7 0
#endif

#if CPU(X86_64)
#define OFFLINE_ASM_X86_64 1
#else
#define OFFLINE_ASM_X86_64 0
#endif

#if CPU(ARM64)
#define OFFLINE_ASM_ARM64 1
#else
#define OFFLINE_ASM_ARM64 0
#endif

#if CPU(ARM64E)
#define OFFLINE_ASM_ARM64E 1
#undef OFFLINE_ASM_ARM64
#define OFFLINE_ASM_ARM64 0 // Pretend that ARM64 and ARM64E are mutually exclusive to please the offlineasm.
#else
#define OFFLINE_ASM_ARM64E 0
#endif

#if CPU(RISCV64)
#define OFFLINE_ASM_RISCV64 1
#else
#define OFFLINE_ASM_RISCV64 0
#endif

#endif // ENABLE(C_LOOP)

#if USE(JSVALUE64)
#define OFFLINE_ASM_JSVALUE64 1
#else
#define OFFLINE_ASM_JSVALUE64 0
#endif

#if USE(BIGINT32)
#define OFFLINE_ASM_BIGINT32 1
#else
#define OFFLINE_ASM_BIGINT32 0
#endif

#if USE(LARGE_TYPED_ARRAYS)
#define OFFLINE_ASM_LARGE_TYPED_ARRAYS 1
#else
#define OFFLINE_ASM_LARGE_TYPED_ARRAYS 0
#endif

#if CPU(ADDRESS64)
#define OFFLINE_ASM_ADDRESS64 1
#else
#define OFFLINE_ASM_ADDRESS64 0
#endif

#if ENABLE(STRUCTURE_ID_WITH_SHIFT)
#define OFFLINE_ASM_STRUCTURE_ID_WITH_SHIFT 1
#else
#define OFFLINE_ASM_STRUCTURE_ID_WITH_SHIFT 0
#endif

#if ASSERT_ENABLED
#define OFFLINE_ASM_ASSERT_ENABLED 1
#else
#define OFFLINE_ASM_ASSERT_ENABLED 0
#endif

#if LLINT_TRACING
#define OFFLINE_ASM_TRACING 1
#else
#define OFFLINE_ASM_TRACING 0
#endif

#define OFFLINE_ASM_GIGACAGE_ENABLED GIGACAGE_ENABLED

#if ENABLE(JIT)
#define OFFLINE_ASM_JIT 1
#else
#define OFFLINE_ASM_JIT 0
#endif

#if ENABLE(JIT_CAGE)
#define OFFLINE_ASM_JIT_CAGE 1
#else
#define OFFLINE_ASM_JIT_CAGE 0
#endif

#if ENABLE(WEBASSEMBLY)
#define OFFLINE_ASM_WEBASSEMBLY 1
#else
#define OFFLINE_ASM_WEBASSEMBLY 0
#endif

#if ENABLE(WEBASSEMBLY_OMGJIT)
#define OFFLINE_ASM_WEBASSEMBLY_OMGJIT 1
#else
#define OFFLINE_ASM_WEBASSEMBLY_OMGJIT 0
#endif

#if ENABLE(WEBASSEMBLY_BBQJIT)
#define OFFLINE_ASM_WEBASSEMBLY_BBQJIT 1
#else
#define OFFLINE_ASM_WEBASSEMBLY_BBQJIT 0
#endif

#if HAVE(FAST_TLS)
#define OFFLINE_ASM_HAVE_FAST_TLS 1
#else
#define OFFLINE_ASM_HAVE_FAST_TLS 0
#endif
