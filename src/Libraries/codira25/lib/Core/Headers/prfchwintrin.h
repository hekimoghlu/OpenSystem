/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 11, 2024.
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
#if !defined(__X86INTRIN_H) && !defined(_MM3DNOW_H_INCLUDED)
#error "Never use <prfchwintrin.h> directly; include <x86intrin.h> instead."
#endif

#ifndef __PRFCHWINTRIN_H
#define __PRFCHWINTRIN_H

#if defined(__cplusplus)
extern "C" {
#endif

/// Loads a memory sequence containing the specified memory address into
///    all data cache levels.
///
///    The cache-coherency state is set to exclusive. Data can be read from
///    and written to the cache line without additional delay.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the \c PREFETCHT0 instruction.
///
/// \param __P
///    A pointer specifying the memory address to be prefetched.
void _m_prefetch(void *__P);

/// Loads a memory sequence containing the specified memory address into
///    the L1 data cache and sets the cache-coherency state to modified.
///
///    This provides a hint to the processor that the cache line will be
///    modified. It is intended for use when the cache line will be written to
///    shortly after the prefetch is performed.
///
///    Note that the effect of this intrinsic is dependent on the processor
///    implementation.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the \c PREFETCHW instruction.
///
/// \param __P
///    A pointer specifying the memory address to be prefetched.
void _m_prefetchw(volatile const void *__P);

#if defined(__cplusplus)
} // extern "C"
#endif

#endif /* __PRFCHWINTRIN_H */
