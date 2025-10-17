/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 28, 2024.
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
#error "Never use <clwbintrin.h> directly; include <immintrin.h> instead."
#endif

#ifndef __CLWBINTRIN_H
#define __CLWBINTRIN_H

/* Define the default attributes for the functions in this file. */
#define __DEFAULT_FN_ATTRS __attribute__((__always_inline__, __nodebug__,  __target__("clwb")))

/// Writes back to memory the cache line (if modified) that contains the
/// linear address specified in \a __p from any level of the cache hierarchy in
/// the cache coherence domain
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the <c> CLWB </c> instruction.
///
/// \param __p
///    A pointer to the memory location used to identify the cache line to be
///    written back.
static __inline__ void __DEFAULT_FN_ATTRS
_mm_clwb(void const *__p) {
  __builtin_ia32_clwb(__p);
}

#undef __DEFAULT_FN_ATTRS

#endif
