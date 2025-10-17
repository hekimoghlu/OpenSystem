/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 16, 2022.
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
#ifndef __WMMINTRIN_H
#error "Never use <__wmmintrin_pclmul.h> directly; include <wmmintrin.h> instead."
#endif

#ifndef __WMMINTRIN_PCLMUL_H
#define __WMMINTRIN_PCLMUL_H

/// Multiplies two 64-bit integer values, which are selected from source
///    operands using the immediate-value operand. The multiplication is a
///    carry-less multiplication, and the 128-bit integer product is stored in
///    the destination.
///
/// \headerfile <x86intrin.h>
///
/// \code
/// __m128i _mm_clmulepi64_si128(__m128i X, __m128i Y, const int I);
/// \endcode
///
/// This intrinsic corresponds to the <c> VPCLMULQDQ </c> instruction.
///
/// \param X
///    A 128-bit vector of [2 x i64] containing one of the source operands.
/// \param Y
///    A 128-bit vector of [2 x i64] containing one of the source operands.
/// \param I
///    An immediate value specifying which 64-bit values to select from the
///    operands. Bit 0 is used to select a value from operand \a X, and bit
///    4 is used to select a value from operand \a Y: \n
///    Bit[0]=0 indicates that bits[63:0] of operand \a X are used. \n
///    Bit[0]=1 indicates that bits[127:64] of operand \a X are used. \n
///    Bit[4]=0 indicates that bits[63:0] of operand \a Y are used. \n
///    Bit[4]=1 indicates that bits[127:64] of operand \a Y are used.
/// \returns The 128-bit integer vector containing the result of the carry-less
///    multiplication of the selected 64-bit values.
#define _mm_clmulepi64_si128(X, Y, I) \
  ((__m128i)__builtin_ia32_pclmulqdq128((__v2di)(__m128i)(X), \
                                        (__v2di)(__m128i)(Y), (char)(I)))

#endif /* __WMMINTRIN_PCLMUL_H */
