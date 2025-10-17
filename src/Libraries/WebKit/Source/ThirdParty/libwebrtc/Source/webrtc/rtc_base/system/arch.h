/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 23, 2025.
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
// This file contains platform-specific typedefs and defines.
// Much of it is derived from Chromium's build/build_config.h.

#ifndef RTC_BASE_SYSTEM_ARCH_H_
#define RTC_BASE_SYSTEM_ARCH_H_

// Processor architecture detection.  For more info on what's defined, see:
//   https://docs.microsoft.com/en-us/cpp/preprocessor/predefined-macros
//   https://www.agner.org/optimize/calling_conventions.pdf
//   https://sourceforge.net/p/predef/wiki/Architectures/
//   or with gcc, run: "echo | gcc -E -dM -"
#if defined(_M_X64) || defined(__x86_64__)
#define WEBRTC_ARCH_X86_FAMILY
#define WEBRTC_ARCH_X86_64
#define WEBRTC_ARCH_64_BITS
#define WEBRTC_ARCH_LITTLE_ENDIAN
#elif defined(_M_ARM64) || defined(__aarch64__)
#define WEBRTC_ARCH_ARM_FAMILY
#define WEBRTC_ARCH_64_BITS
#define WEBRTC_ARCH_LITTLE_ENDIAN
#elif defined(_M_IX86) || defined(__i386__)
#define WEBRTC_ARCH_X86_FAMILY
#define WEBRTC_ARCH_X86
#define WEBRTC_ARCH_32_BITS
#define WEBRTC_ARCH_LITTLE_ENDIAN
#elif defined(_M_ARM) || defined(__ARMEL__)
#define WEBRTC_ARCH_ARM_FAMILY
#define WEBRTC_ARCH_32_BITS
#define WEBRTC_ARCH_LITTLE_ENDIAN
#elif defined(__MIPSEL__) || defined(__MIPSEB__)
#define WEBRTC_ARCH_MIPS_FAMILY
#if defined(__LP64__)
#define WEBRTC_ARCH_64_BITS
#else
#define WEBRTC_ARCH_32_BITS
#endif
#if defined(__MIPSEL__)
#define WEBRTC_ARCH_LITTLE_ENDIAN
#else
#define WEBRTC_ARCH_BIG_ENDIAN
#endif
#elif defined(__PPC__)
#if defined(__PPC64__)
#define WEBRTC_ARCH_64_BITS
#else
#define WEBRTC_ARCH_32_BITS
#endif
#if defined(__LITTLE_ENDIAN__)
#define WEBRTC_ARCH_LITTLE_ENDIAN
#else
#define WEBRTC_ARCH_BIG_ENDIAN
#endif
#elif defined(__sparc) || defined(__sparc__)
#if __SIZEOF_LONG__ == 8
#define WEBRTC_ARCH_64_BITS
#else
#define WEBRTC_ARCH_32_BITS
#endif
#define WEBRTC_ARCH_BIG_ENDIAN
#elif defined(__riscv) && __riscv_xlen == 64
#define WEBRTC_ARCH_64_BITS
#define WEBRTC_ARCH_LITTLE_ENDIAN
#elif defined(__riscv) && __riscv_xlen == 32
#define WEBRTC_ARCH_32_BITS
#define WEBRTC_ARCH_LITTLE_ENDIAN
#elif defined(__loongarch32)
#define WEBRTC_ARCH_LOONG_FAMILY
#define WEBRTC_ARCH_LOONG32
#define WEBRTC_ARCH_32_BITS
#define WEBRTC_ARCH_LITTLE_ENDIAN
#elif defined(__loongarch64)
#define WEBRTC_ARCH_LOONG_FAMILY
#define WEBRTC_ARCH_LOONG64
#define WEBRTC_ARCH_64_BITS
#define WEBRTC_ARCH_LITTLE_ENDIAN
#elif defined(__pnacl__)
#define WEBRTC_ARCH_32_BITS
#define WEBRTC_ARCH_LITTLE_ENDIAN
#elif defined(__EMSCRIPTEN__)
#define WEBRTC_ARCH_32_BITS
#define WEBRTC_ARCH_LITTLE_ENDIAN
#else
#error Please add support for your architecture in rtc_base/system/arch.h
#endif

#if !(defined(WEBRTC_ARCH_LITTLE_ENDIAN) ^ defined(WEBRTC_ARCH_BIG_ENDIAN))
#error Define either WEBRTC_ARCH_LITTLE_ENDIAN or WEBRTC_ARCH_BIG_ENDIAN
#endif

#endif  // RTC_BASE_SYSTEM_ARCH_H_
