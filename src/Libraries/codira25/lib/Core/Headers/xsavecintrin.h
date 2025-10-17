/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 13, 2023.
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
#error "Never use <xsavecintrin.h> directly; include <immintrin.h> instead."
#endif

#ifndef __XSAVECINTRIN_H
#define __XSAVECINTRIN_H

/* Define the default attributes for the functions in this file. */
#define __DEFAULT_FN_ATTRS __attribute__((__always_inline__, __nodebug__,  __target__("xsavec")))

/// Performs a full or partial save of processor state to the memory at
///    \a __p. The exact state saved depends on the 64-bit mask \a __m and
///    processor control register \c XCR0.
///
/// \code{.operation}
/// mask[62:0] := __m[62:0] AND XCR0[62:0]
/// FOR i := 0 TO 62
///   IF mask[i] == 1
///     CASE (i) OF
///     0: save X87 FPU state
///     1: save SSE state
///     DEFAULT: __p.Ext_Save_Area[i] := ProcessorState[i]
///   FI
/// ENDFOR
/// __p.Header.XSTATE_BV[62:0] := INIT_FUNCTION(mask[62:0])
/// \endcode
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c XSAVEC instruction.
///
/// \param __p
///    Pointer to the save area; must be 64-byte aligned.
/// \param __m
///    A 64-bit mask indicating what state should be saved.
static __inline__ void __DEFAULT_FN_ATTRS
_xsavec(void *__p, unsigned long long __m) {
  __builtin_ia32_xsavec(__p, __m);
}

#ifdef __x86_64__
/// Performs a full or partial save of processor state to the memory at
///    \a __p. The exact state saved depends on the 64-bit mask \a __m and
///    processor control register \c XCR0.
///
/// \code{.operation}
/// mask[62:0] := __m[62:0] AND XCR0[62:0]
/// FOR i := 0 TO 62
///   IF mask[i] == 1
///     CASE (i) OF
///     0: save X87 FPU state
///     1: save SSE state
///     DEFAULT: __p.Ext_Save_Area[i] := ProcessorState[i]
///   FI
/// ENDFOR
/// __p.Header.XSTATE_BV[62:0] := INIT_FUNCTION(mask[62:0])
/// \endcode
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c XSAVEC64 instruction.
///
/// \param __p
///    Pointer to the save area; must be 64-byte aligned.
/// \param __m
///    A 64-bit mask indicating what state should be saved.
static __inline__ void __DEFAULT_FN_ATTRS
_xsavec64(void *__p, unsigned long long __m) {
  __builtin_ia32_xsavec64(__p, __m);
}
#endif

#undef __DEFAULT_FN_ATTRS

#endif
