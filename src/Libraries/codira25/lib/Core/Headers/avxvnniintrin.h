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
#error "Never use <avxvnniintrin.h> directly; include <immintrin.h> instead."
#endif

#ifndef __AVXVNNIINTRIN_H
#define __AVXVNNIINTRIN_H

/* Below intrinsics defined in avx512vlvnniintrin.h can be used for AVXVNNI */
/// \fn __m256i _mm256_dpbusd_epi32(__m256i __S, __m256i __A, __m256i __B)
/// \fn __m256i _mm256_dpbusds_epi32(__m256i __S, __m256i __A, __m256i __B)
/// \fn __m256i _mm256_dpwssd_epi32(__m256i __S, __m256i __A, __m256i __B)
/// \fn __m256i _mm256_dpwssds_epi32(__m256i __S, __m256i __A, __m256i __B)
/// \fn __m128i _mm_dpbusd_epi32(__m128i __S, __m128i __A, __m128i __B)
/// \fn __m128i _mm_dpbusds_epi32(__m128i __S, __m128i __A, __m128i __B)
/// \fn __m128i _mm_dpwssd_epi32(__m128i __S, __m128i __A, __m128i __B)
/// \fn __m128i _mm_dpwssds_epi32(__m128i __S, __m128i __A, __m128i __B)

/* Intrinsics with _avx_ prefix are for compatibility with msvc. */
/* Define the default attributes for the functions in this file. */
#define __DEFAULT_FN_ATTRS256 __attribute__((__always_inline__, __nodebug__, __target__("avxvnni"), __min_vector_width__(256)))
#define __DEFAULT_FN_ATTRS128 __attribute__((__always_inline__, __nodebug__, __target__("avxvnni"), __min_vector_width__(128)))

/// Multiply groups of 4 adjacent pairs of unsigned 8-bit integers in \a __A with
/// corresponding signed 8-bit integers in \a __B, producing 4 intermediate signed
/// 16-bit results. Sum these 4 results with the corresponding 32-bit integer
/// in \a __S, and store the packed 32-bit results in DST.
///
/// This intrinsic corresponds to the <c> VPDPBUSD </c> instructions.
///
/// \code{.operation}
///    FOR j := 0 to 7
///      tmp1.word := Signed(ZeroExtend16(__A.byte[4*j]) * SignExtend16(__B.byte[4*j]))
///      tmp2.word := Signed(ZeroExtend16(__A.byte[4*j+1]) * SignExtend16(__B.byte[4*j+1]))
///      tmp3.word := Signed(ZeroExtend16(__A.byte[4*j+2]) * SignExtend16(__B.byte[4*j+2]))
///      tmp4.word := Signed(ZeroExtend16(__A.byte[4*j+3]) * SignExtend16(__B.byte[4*j+3]))
///      DST.dword[j] := __S.dword[j] + tmp1 + tmp2 + tmp3 + tmp4
///    ENDFOR
///    DST[MAX:256] := 0
/// \endcode
static __inline__ __m256i __DEFAULT_FN_ATTRS256
_mm256_dpbusd_avx_epi32(__m256i __S, __m256i __A, __m256i __B)
{
  return (__m256i)__builtin_ia32_vpdpbusd256((__v8si)__S, (__v8si)__A, (__v8si)__B);
}

/// Multiply groups of 4 adjacent pairs of unsigned 8-bit integers in \a __A with
/// corresponding signed 8-bit integers in \a __B, producing 4 intermediate signed
/// 16-bit results. Sum these 4 results with the corresponding 32-bit integer
/// in \a __S using signed saturation, and store the packed 32-bit results in DST.
///
/// This intrinsic corresponds to the <c> VPDPBUSDS </c> instructions.
///
/// \code{.operation}
///    FOR j := 0 to 7
///      tmp1.word := Signed(ZeroExtend16(__A.byte[4*j]) * SignExtend16(__B.byte[4*j]))
///      tmp2.word := Signed(ZeroExtend16(__A.byte[4*j+1]) * SignExtend16(__B.byte[4*j+1]))
///      tmp3.word := Signed(ZeroExtend16(__A.byte[4*j+2]) * SignExtend16(__B.byte[4*j+2]))
///      tmp4.word := Signed(ZeroExtend16(__A.byte[4*j+3]) * SignExtend16(__B.byte[4*j+3]))
///      DST.dword[j] := Saturate32(__S.dword[j] + tmp1 + tmp2 + tmp3 + tmp4)
///    ENDFOR
///    DST[MAX:256] := 0
/// \endcode
static __inline__ __m256i __DEFAULT_FN_ATTRS256
_mm256_dpbusds_avx_epi32(__m256i __S, __m256i __A, __m256i __B)
{
  return (__m256i)__builtin_ia32_vpdpbusds256((__v8si)__S, (__v8si)__A, (__v8si)__B);
}

/// Multiply groups of 2 adjacent pairs of signed 16-bit integers in \a __A with
/// corresponding 16-bit integers in \a __B, producing 2 intermediate signed 32-bit
/// results. Sum these 2 results with the corresponding 32-bit integer in \a __S,
///  and store the packed 32-bit results in DST.
///
/// This intrinsic corresponds to the <c> VPDPWSSD </c> instructions.
///
/// \code{.operation}
///    FOR j := 0 to 7
///      tmp1.dword := SignExtend32(__A.word[2*j]) * SignExtend32(__B.word[2*j])
///      tmp2.dword := SignExtend32(__A.word[2*j+1]) * SignExtend32(__B.word[2*j+1])
///      DST.dword[j] := __S.dword[j] + tmp1 + tmp2
///    ENDFOR
///    DST[MAX:256] := 0
/// \endcode
static __inline__ __m256i __DEFAULT_FN_ATTRS256
_mm256_dpwssd_avx_epi32(__m256i __S, __m256i __A, __m256i __B)
{
  return (__m256i)__builtin_ia32_vpdpwssd256((__v8si)__S, (__v8si)__A, (__v8si)__B);
}

/// Multiply groups of 2 adjacent pairs of signed 16-bit integers in \a __A with
/// corresponding 16-bit integers in \a __B, producing 2 intermediate signed 32-bit
/// results. Sum these 2 results with the corresponding 32-bit integer in \a __S
/// using signed saturation, and store the packed 32-bit results in DST.
///
/// This intrinsic corresponds to the <c> VPDPWSSDS </c> instructions.
///
/// \code{.operation}
///    FOR j := 0 to 7
///      tmp1.dword := SignExtend32(__A.word[2*j]) * SignExtend32(__B.word[2*j])
///      tmp2.dword := SignExtend32(__A.word[2*j+1]) * SignExtend32(__B.word[2*j+1])
///      DST.dword[j] := Saturate32(__S.dword[j] + tmp1 + tmp2)
///    ENDFOR
///    DST[MAX:256] := 0
/// \endcode
static __inline__ __m256i __DEFAULT_FN_ATTRS256
_mm256_dpwssds_avx_epi32(__m256i __S, __m256i __A, __m256i __B)
{
  return (__m256i)__builtin_ia32_vpdpwssds256((__v8si)__S, (__v8si)__A, (__v8si)__B);
}

/// Multiply groups of 4 adjacent pairs of unsigned 8-bit integers in \a __A with
/// corresponding signed 8-bit integers in \a __B, producing 4 intermediate signed
/// 16-bit results. Sum these 4 results with the corresponding 32-bit integer
/// in \a __S, and store the packed 32-bit results in DST.
///
/// This intrinsic corresponds to the <c> VPDPBUSD </c> instructions.
///
/// \code{.operation}
///    FOR j := 0 to 3
///      tmp1.word := Signed(ZeroExtend16(__A.byte[4*j]) * SignExtend16(__B.byte[4*j]))
///      tmp2.word := Signed(ZeroExtend16(__A.byte[4*j+1]) * SignExtend16(__B.byte[4*j+1]))
///      tmp3.word := Signed(ZeroExtend16(__A.byte[4*j+2]) * SignExtend16(__B.byte[4*j+2]))
///      tmp4.word := Signed(ZeroExtend16(__A.byte[4*j+3]) * SignExtend16(__B.byte[4*j+3]))
///      DST.dword[j] := __S.dword[j] + tmp1 + tmp2 + tmp3 + tmp4
///    ENDFOR
///    DST[MAX:128] := 0
/// \endcode
static __inline__ __m128i __DEFAULT_FN_ATTRS128
_mm_dpbusd_avx_epi32(__m128i __S, __m128i __A, __m128i __B)
{
  return (__m128i)__builtin_ia32_vpdpbusd128((__v4si)__S, (__v4si)__A, (__v4si)__B);
}

/// Multiply groups of 4 adjacent pairs of unsigned 8-bit integers in \a __A with
/// corresponding signed 8-bit integers in \a __B, producing 4 intermediate signed
/// 16-bit results. Sum these 4 results with the corresponding 32-bit integer
/// in \a __S using signed saturation, and store the packed 32-bit results in DST.
///
/// This intrinsic corresponds to the <c> VPDPBUSDS </c> instructions.
///
/// \code{.operation}
///    FOR j := 0 to 3
///      tmp1.word := Signed(ZeroExtend16(__A.byte[4*j]) * SignExtend16(__B.byte[4*j]))
///      tmp2.word := Signed(ZeroExtend16(__A.byte[4*j+1]) * SignExtend16(__B.byte[4*j+1]))
///      tmp3.word := Signed(ZeroExtend16(__A.byte[4*j+2]) * SignExtend16(__B.byte[4*j+2]))
///      tmp4.word := Signed(ZeroExtend16(__A.byte[4*j+3]) * SignExtend16(__B.byte[4*j+3]))
///      DST.dword[j] := Saturate32(__S.dword[j] + tmp1 + tmp2 + tmp3 + tmp4)
///    ENDFOR
///    DST[MAX:128] := 0
/// \endcode
static __inline__ __m128i __DEFAULT_FN_ATTRS128
_mm_dpbusds_avx_epi32(__m128i __S, __m128i __A, __m128i __B)
{
  return (__m128i)__builtin_ia32_vpdpbusds128((__v4si)__S, (__v4si)__A, (__v4si)__B);
}

/// Multiply groups of 2 adjacent pairs of signed 16-bit integers in \a __A with
/// corresponding 16-bit integers in \a __B, producing 2 intermediate signed 32-bit
/// results. Sum these 2 results with the corresponding 32-bit integer in \a __S,
/// and store the packed 32-bit results in DST.
///
/// This intrinsic corresponds to the <c> VPDPWSSD </c> instructions.
///
/// \code{.operation}
///    FOR j := 0 to 3
///      tmp1.dword := SignExtend32(__A.word[2*j]) * SignExtend32(__B.word[2*j])
///      tmp2.dword := SignExtend32(__A.word[2*j+1]) * SignExtend32(__B.word[2*j+1])
///      DST.dword[j] := __S.dword[j] + tmp1 + tmp2
///    ENDFOR
///    DST[MAX:128] := 0
/// \endcode
static __inline__ __m128i __DEFAULT_FN_ATTRS128
_mm_dpwssd_avx_epi32(__m128i __S, __m128i __A, __m128i __B)
{
  return (__m128i)__builtin_ia32_vpdpwssd128((__v4si)__S, (__v4si)__A, (__v4si)__B);
}

/// Multiply groups of 2 adjacent pairs of signed 16-bit integers in \a __A with
/// corresponding 16-bit integers in \a __B, producing 2 intermediate signed 32-bit
/// results. Sum these 2 results with the corresponding 32-bit integer in \a __S
/// using signed saturation, and store the packed 32-bit results in DST.
///
/// This intrinsic corresponds to the <c> VPDPWSSDS </c> instructions.
///
/// \code{.operation}
///    FOR j := 0 to 3
///      tmp1.dword := SignExtend32(__A.word[2*j]) * SignExtend32(__B.word[2*j])
///      tmp2.dword := SignExtend32(__A.word[2*j+1]) * SignExtend32(__B.word[2*j+1])
///      DST.dword[j] := Saturate32(__S.dword[j] + tmp1 + tmp2)
///    ENDFOR
///    DST[MAX:128] := 0
/// \endcode
static __inline__ __m128i __DEFAULT_FN_ATTRS128
_mm_dpwssds_avx_epi32(__m128i __S, __m128i __A, __m128i __B)
{
  return (__m128i)__builtin_ia32_vpdpwssds128((__v4si)__S, (__v4si)__A, (__v4si)__B);
}

#undef __DEFAULT_FN_ATTRS128
#undef __DEFAULT_FN_ATTRS256

#endif // __AVXVNNIINTRIN_H
