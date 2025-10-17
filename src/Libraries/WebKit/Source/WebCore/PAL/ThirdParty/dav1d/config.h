/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 15, 2025.
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

#define ARCH_AARCH64 0
#if defined(__arm64__) || defined(__aarch64__)
#undef ARCH_AARCH64
#define ARCH_AARCH64 1
#endif

#define ARCH_ARM 0
#if defined(arm) || defined(__arm__) || defined(ARM) || defined(_ARM_)
#undef ARCH_ARM
#define ARCH_ARM 1
#endif

#define ARCH_PPC64LE 0

#define ARCH_X86 0

#define ARCH_X86_32 0
#if defined(__i386__) || defined(i386) || defined(_M_IX86) || defined(_X86_) || defined(__THW_INTEL)
#undef ARCH_X86_32
#define ARCH_X86_32 1
#undef ARCH_X86
#define ARCH_X86 1
#endif

#define ARCH_X86_64 0
#if defined(__x86_64__) || defined(_M_X64)
#undef ARCH_X86_64
#define ARCH_X86_64 1
#undef ARCH_X86
#define ARCH_X86 1
#endif

#define CONFIG_16BPC 1

#define CONFIG_8BPC 1

#define CONFIG_LOG 0

#define ENDIANNESS_BIG 0

#define HAVE_ASM 0

#define HAVE_CLOCK_GETTIME 1

#define HAVE_POSIX_MEMALIGN 1

#define HAVE_UNISTD_H 1
