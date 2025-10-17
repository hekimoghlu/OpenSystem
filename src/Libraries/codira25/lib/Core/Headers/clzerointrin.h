/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 17, 2023.
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
#ifndef __X86INTRIN_H
#error "Never use <clzerointrin.h> directly; include <x86intrin.h> instead."
#endif

#ifndef __CLZEROINTRIN_H
#define __CLZEROINTRIN_H

/* Define the default attributes for the functions in this file. */
#define __DEFAULT_FN_ATTRS \
  __attribute__((__always_inline__, __nodebug__,  __target__("clzero")))

/// Zeroes out the cache line for the address \a __line. This uses a
///    non-temporal store. Calling \c _mm_sfence() afterward might be needed
///    to enforce ordering.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the \c CLZERO instruction.
///
/// \param __line
///    An address within the cache line to zero out.
static __inline__ void __DEFAULT_FN_ATTRS
_mm_clzero (void * __line)
{
  __builtin_ia32_clzero ((void *)__line);
}

#undef __DEFAULT_FN_ATTRS

#endif /* __CLZEROINTRIN_H */
