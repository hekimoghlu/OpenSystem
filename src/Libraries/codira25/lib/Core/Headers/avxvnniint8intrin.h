/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 2, 2023.
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
#error                                                                         \
    "Never use <avxvnniint8intrin.h> directly; include <immintrin.h> instead."
#endif

#ifndef __AVXVNNIINT8INTRIN_H
#define __AVXVNNIINT8INTRIN_H

/// Multiply groups of 4 adjacent pairs of signed 8-bit integers in \a __A with
///    corresponding signed 8-bit integers in \a __B, producing 4 intermediate
///    signed 16-bit results. Sum these 4 results with the corresponding
///    32-bit integer in \a __W, and store the packed 32-bit results in \a dst.
///
/// \headerfile <x86intrin.h>
///
/// \code
/// _mm_dpbssd_epi32(__m128i __W, __m128i __A, __m128i __B);
/// \endcode
///
/// This intrinsic corresponds to the \c VPDPBSSD instruction.
///
/// \param __A
///    A 128-bit vector of [16 x char].
/// \param __B
///    A 128-bit vector of [16 x char].
/// \returns
///    A 128-bit vector of [4 x int].
///
/// \code{.operation}
/// FOR j := 0 to 3
/// 	tmp1.word := SignExtend16(__A.byte[4*j]) * SignExtend16(__B.byte[4*j])
/// 	tmp2.word := SignExtend16(__A.byte[4*j+1]) * SignExtend16(__B.byte[4*j+1])
/// 	tmp3.word := SignExtend16(__A.byte[4*j+2]) * SignExtend16(__B.byte[4*j+2])
/// 	tmp4.word := SignExtend16(__A.byte[4*j+3]) * SignExtend16(__B.byte[4*j+3])
/// 	dst.dword[j] := __W.dword[j] + tmp1 + tmp2 + tmp3 + tmp4
/// ENDFOR
/// dst[MAX:128] := 0
/// \endcode
#define _mm_dpbssd_epi32(__W, __A, __B)                                        \
  ((__m128i)__builtin_ia32_vpdpbssd128((__v4si)(__W), (__v4si)(__A),           \
                                       (__v4si)(__B)))

/// Multiply groups of 4 adjacent pairs of signed 8-bit integers in \a __A with
///    corresponding signed 8-bit integers in \a __B, producing 4 intermediate
///    signed 16-bit results. Sum these 4 results with the corresponding
///    32-bit integer in \a __W, and store the packed 32-bit results in \a dst.
///
/// \headerfile <x86intrin.h>
///
/// \code
/// _mm256_dpbssd_epi32(__m256i __W, __m256i __A, __m256i __B);
/// \endcode
///
/// This intrinsic corresponds to the \c VPDPBSSD instruction.
///
/// \param __A
///    A 256-bit vector of [32 x char].
/// \param __B
///    A 256-bit vector of [32 x char].
/// \returns
///    A 256-bit vector of [8 x int].
///
/// \code{.operation}
/// FOR j := 0 to 7
/// 	tmp1.word := SignExtend16(__A.byte[4*j]) * SignExtend16(__B.byte[4*j])
/// 	tmp2.word := SignExtend16(__A.byte[4*j+1]) * SignExtend16(__B.byte[4*j+1])
/// 	tmp3.word := SignExtend16(__A.byte[4*j+2]) * SignExtend16(__B.byte[4*j+2])
/// 	tmp4.word := SignExtend16(__A.byte[4*j+3]) * SignExtend16(__B.byte[4*j+3])
/// 	dst.dword[j] := __W.dword[j] + tmp1 + tmp2 + tmp3 + tmp4
/// ENDFOR
/// dst[MAX:256] := 0
/// \endcode
#define _mm256_dpbssd_epi32(__W, __A, __B)                                     \
  ((__m256i)__builtin_ia32_vpdpbssd256((__v8si)(__W), (__v8si)(__A),           \
                                       (__v8si)(__B)))

/// Multiply groups of 4 adjacent pairs of signed 8-bit integers in \a __A with
///    corresponding signed 8-bit integers in \a __B, producing 4 intermediate
///    signed 16-bit results. Sum these 4 results with the corresponding
///    32-bit integer in \a __W with signed saturation, and store the packed
///    32-bit results in \a dst.
///
/// \headerfile <x86intrin.h>
///
/// \code
/// _mm_dpbssds_epi32( __m128i __W, __m128i __A, __m128i __B);
/// \endcode
///
/// This intrinsic corresponds to the \c VPDPBSSD instruction.
///
/// \param __A
///    A 128-bit vector of [16 x char].
/// \param __B
///    A 128-bit vector of [16 x char].
/// \returns
///    A 128-bit vector of [4 x int].
///
/// \code{.operation}
/// FOR j := 0 to 3
/// 	tmp1.word := SignExtend16(__A.byte[4*j]) * SignExtend16(__B.byte[4*j])
/// 	tmp2.word := SignExtend16(__A.byte[4*j+1]) * SignExtend16(__B.byte[4*j+1])
/// 	tmp3.word := SignExtend16(__A.byte[4*j+2]) * SignExtend16(__B.byte[4*j+2])
/// 	tmp4.word := SignExtend16(__A.byte[4*j+3]) * SignExtend16(__B.byte[4*j+3])
/// 	dst.dword[j] := SIGNED_DWORD_SATURATE(__W.dword[j] + tmp1 + tmp2 + tmp3 + tmp4)
/// ENDFOR
/// dst[MAX:128] := 0
/// \endcode
#define _mm_dpbssds_epi32(__W, __A, __B)                                       \
  ((__m128i)__builtin_ia32_vpdpbssds128((__v4si)(__W), (__v4si)(__A),          \
                                        (__v4si)(__B)))

/// Multiply groups of 4 adjacent pairs of signed 8-bit integers in \a __A with
///    corresponding signed 8-bit integers in \a __B, producing 4 intermediate
///    signed 16-bit results. Sum these 4 results with the corresponding
///    32-bit integer in \a __W with signed saturation, and store the packed
///    32-bit results in \a dst.
///
/// \headerfile <x86intrin.h>
///
/// \code
/// _mm256_dpbssds_epi32(__m256i __W, __m256i __A, __m256i __B);
/// \endcode
///
/// This intrinsic corresponds to the \c VPDPBSSD instruction.
///
/// \param __A
///    A 256-bit vector of [32 x char].
/// \param __B
///    A 256-bit vector of [32 x char].
/// \returns
///    A 256-bit vector of [8 x int].
///
/// \code{.operation}
/// FOR j := 0 to 7
/// 	tmp1.word := SignExtend16(__A.byte[4*j]) * SignExtend16(__B.byte[4*j])
/// 	tmp2.word := SignExtend16(__A.byte[4*j+1]) * SignExtend16(__B.byte[4*j+1])
/// 	tmp3.word := SignExtend16(__A.byte[4*j+2]) * SignExtend16(__B.byte[4*j+2])
/// 	tmp4.word := SignExtend16(__A.byte[4*j+3]) * SignExtend16(__B.byte[4*j+3])
/// 	dst.dword[j] := SIGNED_DWORD_SATURATE(__W.dword[j] + tmp1 + tmp2 + tmp3 + tmp4)
/// ENDFOR
/// dst[MAX:256] := 0
/// \endcode
#define _mm256_dpbssds_epi32(__W, __A, __B)                                    \
  ((__m256i)__builtin_ia32_vpdpbssds256((__v8si)(__W), (__v8si)(__A),          \
                                        (__v8si)(__B)))

/// Multiply groups of 4 adjacent pairs of signed 8-bit integers in \a __A with
///    corresponding unsigned 8-bit integers in \a __B, producing 4 intermediate
///    signed 16-bit results. Sum these 4 results with the corresponding
///    32-bit integer in \a __W, and store the packed 32-bit results in \a dst.
///
/// \headerfile <x86intrin.h>
///
/// \code
/// _mm_dpbsud_epi32(__m128i __W, __m128i __A, __m128i __B);
/// \endcode
///
/// This intrinsic corresponds to the \c VPDPBSSD instruction.
///
/// \param __A
///    A 128-bit vector of [16 x char].
/// \param __B
///    A 128-bit vector of [16 x unsigned char].
/// \returns
///    A 128-bit vector of [4 x int].
///
/// \code{.operation}
/// FOR j := 0 to 3
/// 	tmp1.word := Signed(SignExtend16(__A.byte[4*j]) * ZeroExtend16(__B.byte[4*j]))
/// 	tmp2.word := Signed(SignExtend16(__A.byte[4*j+1]) * ZeroExtend16(__B.byte[4*j+1]))
/// 	tmp3.word := Signed(SignExtend16(__A.byte[4*j+2]) * ZeroExtend16(__B.byte[4*j+2]))
/// 	tmp4.word := Signed(SignExtend16(__A.byte[4*j+3]) * ZeroExtend16(__B.byte[4*j+3]))
/// 	dst.dword[j] := __W.dword[j] + tmp1 + tmp2 + tmp3 + tmp4
/// ENDFOR
/// dst[MAX:128] := 0
/// \endcode
#define _mm_dpbsud_epi32(__W, __A, __B)                                        \
  ((__m128i)__builtin_ia32_vpdpbsud128((__v4si)(__W), (__v4si)(__A),           \
                                       (__v4si)(__B)))

/// Multiply groups of 4 adjacent pairs of signed 8-bit integers in \a __A with
///    corresponding unsigned 8-bit integers in \a __B, producing 4 intermediate
///    signed 16-bit results. Sum these 4 results with the corresponding
///    32-bit integer in \a __W, and store the packed 32-bit results in \a dst.
///
/// \headerfile <x86intrin.h>
///
/// \code
/// _mm256_dpbsud_epi32(__m256i __W, __m256i __A, __m256i __B);
/// \endcode
///
/// This intrinsic corresponds to the \c VPDPBSSD instruction.
///
/// \param __A
///    A 256-bit vector of [32 x char].
/// \param __B
///    A 256-bit vector of [32 x unsigned char].
/// \returns
///    A 256-bit vector of [8 x int].
///
/// \code{.operation}
/// FOR j := 0 to 7
/// 	tmp1.word := Signed(SignExtend16(__A.byte[4*j]) * ZeroExtend16(__B.byte[4*j]))
/// 	tmp2.word := Signed(SignExtend16(__A.byte[4*j+1]) * ZeroExtend16(__B.byte[4*j+1]))
/// 	tmp3.word := Signed(SignExtend16(__A.byte[4*j+2]) * ZeroExtend16(__B.byte[4*j+2]))
/// 	tmp4.word := Signed(SignExtend16(__A.byte[4*j+3]) * ZeroExtend16(__B.byte[4*j+3]))
/// 	dst.dword[j] := __W.dword[j] + tmp1 + tmp2 + tmp3 + tmp4
/// ENDFOR
/// dst[MAX:256] := 0
/// \endcode
#define _mm256_dpbsud_epi32(__W, __A, __B)                                     \
  ((__m256i)__builtin_ia32_vpdpbsud256((__v8si)(__W), (__v8si)(__A),           \
                                       (__v8si)(__B)))

/// Multiply groups of 4 adjacent pairs of signed 8-bit integers in \a __A with
///    corresponding unsigned 8-bit integers in \a __B, producing 4 intermediate
///    signed 16-bit results. Sum these 4 results with the corresponding
///    32-bit integer in \a __W with signed saturation, and store the packed
///    32-bit results in \a dst.
///
/// \headerfile <x86intrin.h>
///
/// \code
/// _mm_dpbsuds_epi32( __m128i __W, __m128i __A, __m128i __B);
/// \endcode
///
/// This intrinsic corresponds to the \c VPDPBSSD instruction.
///
/// \param __A
///    A 128-bit vector of [16 x char].
/// \param __B
///    A 128-bit vector of [16 x unsigned char].
/// \returns
///    A 128-bit vector of [4 x int].
///
/// \code{.operation}
/// FOR j := 0 to 3
/// 	tmp1.word := Signed(SignExtend16(__A.byte[4*j]) * ZeroExtend16(__B.byte[4*j]))
/// 	tmp2.word := Signed(SignExtend16(__A.byte[4*j+1]) * ZeroExtend16(__B.byte[4*j+1]))
/// 	tmp3.word := Signed(SignExtend16(__A.byte[4*j+2]) * ZeroExtend16(__B.byte[4*j+2]))
/// 	tmp4.word := Signed(SignExtend16(__A.byte[4*j+3]) * ZeroExtend16(__B.byte[4*j+3]))
/// 	dst.dword[j] := SIGNED_DWORD_SATURATE(__W.dword[j] + tmp1 + tmp2 + tmp3 + tmp4)
/// ENDFOR
/// dst[MAX:128] := 0
/// \endcode
#define _mm_dpbsuds_epi32(__W, __A, __B)                                       \
  ((__m128i)__builtin_ia32_vpdpbsuds128((__v4si)(__W), (__v4si)(__A),          \
                                        (__v4si)(__B)))

/// Multiply groups of 4 adjacent pairs of signed 8-bit integers in \a __A with
///    corresponding unsigned 8-bit integers in \a __B, producing 4 intermediate
///    signed 16-bit results. Sum these 4 results with the corresponding
///    32-bit integer in \a __W with signed saturation, and store the packed
///    32-bit results in \a dst.
///
/// \headerfile <x86intrin.h>
///
/// \code
/// _mm256_dpbsuds_epi32(__m256i __W, __m256i __A, __m256i __B);
/// \endcode
///
/// This intrinsic corresponds to the \c VPDPBSSD instruction.
///
/// \param __A
///    A 256-bit vector of [32 x char].
/// \param __B
///    A 256-bit vector of [32 x unsigned char].
/// \returns
///    A 256-bit vector of [8 x int].
///
/// \code{.operation}
/// FOR j := 0 to 7
/// 	tmp1.word := Signed(SignExtend16(__A.byte[4*j]) * ZeroExtend16(__B.byte[4*j]))
/// 	tmp2.word := Signed(SignExtend16(__A.byte[4*j+1]) * ZeroExtend16(__B.byte[4*j+1]))
/// 	tmp3.word := Signed(SignExtend16(__A.byte[4*j+2]) * ZeroExtend16(__B.byte[4*j+2]))
/// 	tmp4.word := Signed(SignExtend16(__A.byte[4*j+3]) * ZeroExtend16(__B.byte[4*j+3]))
/// 	dst.dword[j] := SIGNED_DWORD_SATURATE(__W.dword[j] + tmp1 + tmp2 + tmp3 + tmp4)
/// ENDFOR
/// dst[MAX:256] := 0
/// \endcode
#define _mm256_dpbsuds_epi32(__W, __A, __B)                                    \
  ((__m256i)__builtin_ia32_vpdpbsuds256((__v8si)(__W), (__v8si)(__A),          \
                                        (__v8si)(__B)))

/// Multiply groups of 4 adjacent pairs of unsigned 8-bit integers in \a __A with
///    corresponding unsigned 8-bit integers in \a __B, producing 4 intermediate
///    signed 16-bit results. Sum these 4 results with the corresponding
///    32-bit integer in \a __W, and store the packed 32-bit results in \a dst.
///
/// \headerfile <x86intrin.h>
///
/// \code
/// _mm_dpbuud_epi32(__m128i __W, __m128i __A, __m128i __B);
/// \endcode
///
/// This intrinsic corresponds to the \c VPDPBSSD instruction.
///
/// \param __A
///    A 128-bit vector of [16 x unsigned char].
/// \param __B
///    A 128-bit vector of [16 x unsigned char].
/// \returns
///    A 128-bit vector of [4 x int].
///
/// \code{.operation}
/// FOR j := 0 to 3
/// 	tmp1.word := ZeroExtend16(__A.byte[4*j]) * ZeroExtend16(__B.byte[4*j])
/// 	tmp2.word := ZeroExtend16(__A.byte[4*j+1]) * ZeroExtend16(__B.byte[4*j+1])
/// 	tmp3.word := ZeroExtend16(__A.byte[4*j+2]) * ZeroExtend16(__B.byte[4*j+2])
/// 	tmp4.word := ZeroExtend16(__A.byte[4*j+3]) * ZeroExtend16(__B.byte[4*j+3])
/// 	dst.dword[j] := __W.dword[j] + tmp1 + tmp2 + tmp3 + tmp4
/// ENDFOR
/// dst[MAX:128] := 0
/// \endcode
#define _mm_dpbuud_epi32(__W, __A, __B)                                        \
  ((__m128i)__builtin_ia32_vpdpbuud128((__v4si)(__W), (__v4si)(__A),           \
                                       (__v4si)(__B)))

/// Multiply groups of 4 adjacent pairs of unsigned 8-bit integers in \a __A with
///    corresponding unsigned 8-bit integers in \a __B, producing 4 intermediate
///    signed 16-bit results. Sum these 4 results with the corresponding
///    32-bit integer in \a __W, and store the packed 32-bit results in \a dst.
///
/// \headerfile <x86intrin.h>
///
/// \code
/// _mm256_dpbuud_epi32(__m256i __W, __m256i __A, __m256i __B);
/// \endcode
///
/// This intrinsic corresponds to the \c VPDPBSSD instruction.
///
/// \param __A
///    A 256-bit vector of [32 x unsigned char].
/// \param __B
///    A 256-bit vector of [32 x unsigned char].
/// \returns
///    A 256-bit vector of [8 x int].
///
/// \code{.operation}
/// FOR j := 0 to 7
/// 	tmp1.word := ZeroExtend16(__A.byte[4*j]) * ZeroExtend16(__B.byte[4*j])
/// 	tmp2.word := ZeroExtend16(__A.byte[4*j+1]) * ZeroExtend16(__B.byte[4*j+1])
/// 	tmp3.word := ZeroExtend16(__A.byte[4*j+2]) * ZeroExtend16(__B.byte[4*j+2])
/// 	tmp4.word := ZeroExtend16(__A.byte[4*j+3]) * ZeroExtend16(__B.byte[4*j+3])
/// 	dst.dword[j] := __W.dword[j] + tmp1 + tmp2 + tmp3 + tmp4
/// ENDFOR
/// dst[MAX:256] := 0
/// \endcode
#define _mm256_dpbuud_epi32(__W, __A, __B)                                     \
  ((__m256i)__builtin_ia32_vpdpbuud256((__v8si)(__W), (__v8si)(__A),           \
                                       (__v8si)(__B)))

/// Multiply groups of 4 adjacent pairs of unsigned 8-bit integers in \a __A with
///    corresponding unsigned 8-bit integers in \a __B, producing 4 intermediate
///    signed 16-bit results. Sum these 4 results with the corresponding
///    32-bit integer in \a __W with signed saturation, and store the packed
///    32-bit results in \a dst.
///
/// \headerfile <x86intrin.h>
///
/// \code
/// _mm_dpbuuds_epi32( __m128i __W, __m128i __A, __m128i __B);
/// \endcode
///
/// This intrinsic corresponds to the \c VPDPBUUDS instruction.
///
/// \param __A
///    A 128-bit vector of [16 x unsigned char].
/// \param __B
///    A 128-bit vector of [16 x unsigned char].
/// \returns
///    A 128-bit vector of [4 x int].
///
/// \code{.operation}
/// FOR j := 0 to 3
/// 	tmp1.word := ZeroExtend16(__A.byte[4*j]) * ZeroExtend16(__B.byte[4*j])
/// 	tmp2.word := ZeroExtend16(__A.byte[4*j+1]) * ZeroExtend16(__B.byte[4*j+1])
/// 	tmp3.word := ZeroExtend16(__A.byte[4*j+2]) * ZeroExtend16(__B.byte[4*j+2])
/// 	tmp4.word := ZeroExtend16(__A.byte[4*j+3]) * ZeroExtend16(__B.byte[4*j+3])
/// 	dst.dword[j] := UNSIGNED_DWORD_SATURATE(__W.dword[j] + tmp1 + tmp2 + tmp3 + tmp4)
/// ENDFOR
/// dst[MAX:128] := 0
/// \endcode
#define _mm_dpbuuds_epi32(__W, __A, __B)                                       \
  ((__m128i)__builtin_ia32_vpdpbuuds128((__v4si)(__W), (__v4si)(__A),          \
                                        (__v4si)(__B)))

///    corresponding unsigned 8-bit integers in \a __B, producing 4 intermediate
///    signed 16-bit results. Sum these 4 results with the corresponding
///    32-bit integer in \a __W with signed saturation, and store the packed
///    32-bit results in \a dst.
///
/// \headerfile <x86intrin.h>
///
/// \code
/// _mm256_dpbuuds_epi32(__m256i __W, __m256i __A, __m256i __B);
/// \endcode
///
/// This intrinsic corresponds to the \c VPDPBUUDS instruction.
///
/// \param __A
///    A 256-bit vector of [32 x unsigned char].
/// \param __B
///    A 256-bit vector of [32 x unsigned char].
/// \returns
///    A 256-bit vector of [8 x int].
///
/// \code{.operation}
/// FOR j := 0 to 7
/// 	tmp1.word := ZeroExtend16(__A.byte[4*j]) * ZeroExtend16(__B.byte[4*j])
/// 	tmp2.word := ZeroExtend16(__A.byte[4*j+1]) * ZeroExtend16(__B.byte[4*j+1])
/// 	tmp3.word := ZeroExtend16(__A.byte[4*j+2]) * ZeroExtend16(__B.byte[4*j+2])
/// 	tmp4.word := ZeroExtend16(__A.byte[4*j+3]) * ZeroExtend16(__B.byte[4*j+3])
/// 	dst.dword[j] := UNSIGNED_DWORD_SATURATE(__W.dword[j] + tmp1 + tmp2 + tmp3 + tmp4)
/// ENDFOR
/// dst[MAX:256] := 0
/// \endcode
#define _mm256_dpbuuds_epi32(__W, __A, __B)                                    \
  ((__m256i)__builtin_ia32_vpdpbuuds256((__v8si)(__W), (__v8si)(__A),          \
                                        (__v8si)(__B)))

#endif // __AVXVNNIINT8INTRIN_H
