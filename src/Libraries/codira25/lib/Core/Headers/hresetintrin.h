/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 7, 2022.
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
#ifndef __X86GPRINTRIN_H
#error "Never use <hresetintrin.h> directly; include <x86gprintrin.h> instead."
#endif

#ifndef __HRESETINTRIN_H
#define __HRESETINTRIN_H

#if __has_extension(gnu_asm)

/* Define the default attributes for the functions in this file. */
#define __DEFAULT_FN_ATTRS \
  __attribute__((__always_inline__, __nodebug__, __target__("hreset")))

/// Provides a hint to the processor to selectively reset the prediction
///    history of the current logical processor specified by a 32-bit integer
///    value \a __eax.
///
/// This intrinsic corresponds to the <c> HRESET </c> instruction.
///
/// \code{.operation}
///    IF __eax == 0
///      // nop
///    ELSE
///      FOR i := 0 to 31
///        IF __eax[i]
///          ResetPredictionFeature(i)
///        FI
///      ENDFOR
///    FI
/// \endcode
static __inline void __DEFAULT_FN_ATTRS
_hreset(int __eax)
{
  __asm__ ("hreset $0" :: "a"(__eax));
}

#undef __DEFAULT_FN_ATTRS

#endif /* __has_extension(gnu_asm) */

#endif /* __HRESETINTRIN_H */
