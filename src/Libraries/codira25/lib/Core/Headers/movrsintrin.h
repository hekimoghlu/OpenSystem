/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 9, 2024.
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
#ifndef __IMMINTRIN_H
#error "Never use <movrsintrin.h> directly; include <immintrin.h> instead."
#endif // __IMMINTRIN_H

#ifndef __MOVRSINTRIN_H
#define __MOVRSINTRIN_H

#define __DEFAULT_FN_ATTRS                                                     \
  __attribute__((__always_inline__, __nodebug__, __target__("movrs")))

#ifdef __x86_64__
static __inline__ char __DEFAULT_FN_ATTRS _movrs_i8(const void *__A) {
  return (char)__builtin_ia32_movrsqi((const void *)__A);
}

static __inline__ short __DEFAULT_FN_ATTRS _movrs_i16(const void *__A) {
  return (short)__builtin_ia32_movrshi((const void *)__A);
}

static __inline__ int __DEFAULT_FN_ATTRS _movrs_i32(const void *__A) {
  return (int)__builtin_ia32_movrssi((const void *)__A);
}

static __inline__ long long __DEFAULT_FN_ATTRS _movrs_i64(const void *__A) {
  return (long long)__builtin_ia32_movrsdi((const void *)__A);
}
#endif // __x86_64__

// Loads a memory sequence containing the specified memory address into
/// the L3 data cache. Data will be shared (read/written) to by requesting
/// core and other cores.
///
/// Note that the effect of this intrinsic is dependent on the processor
/// implementation.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the \c PREFETCHRS instruction.
///
/// \param __P
///    A pointer specifying the memory address to be prefetched.
static __inline__ void __DEFAULT_FN_ATTRS
_m_prefetchrs(volatile const void *__P) {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wcast-qual"
  __builtin_ia32_prefetchrs((const void *)__P);
#pragma clang diagnostic pop
}

#undef __DEFAULT_FN_ATTRS
#endif // __MOVRSINTRIN_H
