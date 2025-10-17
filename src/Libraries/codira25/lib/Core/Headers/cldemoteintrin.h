/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 10, 2024.
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
#if !defined __X86INTRIN_H && !defined __IMMINTRIN_H
#error "Never use <cldemoteintrin.h> directly; include <x86intrin.h> instead."
#endif

#ifndef __CLDEMOTEINTRIN_H
#define __CLDEMOTEINTRIN_H

/* Define the default attributes for the functions in this file. */
#define __DEFAULT_FN_ATTRS \
  __attribute__((__always_inline__, __nodebug__,  __target__("cldemote")))

/// Hint to hardware that the cache line that contains \p __P should be demoted
/// from the cache closest to the processor core to a level more distant from
/// the processor core.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the <c> CLDEMOTE </c> instruction.
static __inline__ void __DEFAULT_FN_ATTRS
_cldemote(const void * __P) {
  __builtin_ia32_cldemote(__P);
}

#define _mm_cldemote(p) _cldemote(p)
#undef __DEFAULT_FN_ATTRS

#endif
