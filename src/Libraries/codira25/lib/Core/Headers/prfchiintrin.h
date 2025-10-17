/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 17, 2022.
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
#ifndef __PRFCHIINTRIN_H
#define __PRFCHIINTRIN_H

#ifdef __x86_64__

/* Define the default attributes for the functions in this file. */
#define __DEFAULT_FN_ATTRS                                                     \
  __attribute__((__always_inline__, __nodebug__, __target__("prefetchi")))

/// Loads an instruction sequence containing the specified memory address into
///    all level cache.
///
///    Note that the effect of this intrinsic is dependent on the processor
///    implementation.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the \c PREFETCHIT0 instruction.
///
/// \param __P
///    A pointer specifying the memory address to be prefetched.
static __inline__ void __DEFAULT_FN_ATTRS
_m_prefetchit0(volatile const void *__P) {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wcast-qual"
  __builtin_ia32_prefetchi((const void *)__P, 3 /* _MM_HINT_T0 */);
#pragma clang diagnostic pop
}

/// Loads an instruction sequence containing the specified memory address into
///    all but the first-level cache.
///
///    Note that the effect of this intrinsic is dependent on the processor
///    implementation.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the \c PREFETCHIT1 instruction.
///
/// \param __P
///    A pointer specifying the memory address to be prefetched.
static __inline__ void __DEFAULT_FN_ATTRS
_m_prefetchit1(volatile const void *__P) {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wcast-qual"
  __builtin_ia32_prefetchi((const void *)__P, 2 /* _MM_HINT_T1 */);
#pragma clang diagnostic pop
}
#endif /* __x86_64__ */
#undef __DEFAULT_FN_ATTRS

#endif /* __PRFCHWINTRIN_H */
