/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 14, 2022.
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
#error "Never use <clflushoptintrin.h> directly; include <immintrin.h> instead."
#endif

#ifndef __CLFLUSHOPTINTRIN_H
#define __CLFLUSHOPTINTRIN_H

/* Define the default attributes for the functions in this file. */
#define __DEFAULT_FN_ATTRS __attribute__((__always_inline__, __nodebug__,  __target__("clflushopt")))

/// Invalidates all levels of the cache hierarchy and flushes modified data to
///    memory for the cache line specified by the address \a __m.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c CLFLUSHOPT instruction.
///
/// \param __m
///    An address within the cache line to flush and invalidate.
static __inline__ void __DEFAULT_FN_ATTRS
_mm_clflushopt(void const * __m) {
  __builtin_ia32_clflushopt(__m);
}

#undef __DEFAULT_FN_ATTRS

#endif
