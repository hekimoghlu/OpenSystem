/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 11, 2024.
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
#ifndef PAS_CONFIG_PREFIX_H
#define PAS_CONFIG_PREFIX_H

#include "pas_platform.h"

#if defined(ENABLE_PAS_TESTING)
#define __PAS_ENABLE_TESTING 1
#else
#define __PAS_ENABLE_TESTING 0
#endif

#if (defined(__arm64__) && defined(__APPLE__)) || defined(__aarch64__) || defined(__arm64e__)
#define __PAS_ARM64 1
#if defined(__arm64e__)
#define __PAS_ARM64E 1
#else
#define __PAS_ARM64E 0
#endif
#else
#define __PAS_ARM64 0
#define __PAS_ARM64E 0
#endif

#if (defined(arm) || defined(__arm__) || defined(ARM) || defined(_ARM_)) && !__PAS_ARM64
#define __PAS_ARM32 1
#else
#define __PAS_ARM32 0
#endif

#define __PAS_ARM (!!__PAS_ARM64 || !!__PAS_ARM32)

#if defined(__i386__) || defined(i386) || defined(_M_IX86) || defined(_X86_) || defined(__THW_INTEL)
#define __PAS_X86 1
#else
#define __PAS_X86 0
#endif

#if defined(__x86_64__) || defined(_M_X64)
#define __PAS_X86_64 1
#else
#define __PAS_X86_64 0
#endif

#if defined(__riscv)
#define __PAS_RISCV 1
#else
#define __PAS_RISCV 0
#endif

#endif /* PAS_CONFIG_PREFIX_H */
