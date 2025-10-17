/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 13, 2023.
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
#if !defined(X86CPU_H)
# define X86CPU_H

# if defined(OPUS_X86_MAY_HAVE_SSE)
#  define MAY_HAVE_SSE(name) name ## _sse
# else
#  define MAY_HAVE_SSE(name) name ## _c
# endif

# if defined(OPUS_X86_MAY_HAVE_SSE2)
#  define MAY_HAVE_SSE2(name) name ## _sse2
# else
#  define MAY_HAVE_SSE2(name) name ## _c
# endif

# if defined(OPUS_X86_MAY_HAVE_SSE4_1)
#  define MAY_HAVE_SSE4_1(name) name ## _sse4_1
# else
#  define MAY_HAVE_SSE4_1(name) name ## _c
# endif

# if defined(OPUS_X86_MAY_HAVE_AVX)
#  define MAY_HAVE_AVX(name) name ## _avx
# else
#  define MAY_HAVE_AVX(name) name ## _c
# endif

# if defined(OPUS_HAVE_RTCD)
int opus_select_arch(void);
# endif

/*MOVD should not impose any alignment restrictions, but the C standard does,
   and UBSan will report errors if we actually make unaligned accesses.
  Use this to work around those restrictions (which should hopefully all get
   optimized to a single MOVD instruction).*/
#define OP_LOADU_EPI32(x) \
  (int)((*(unsigned char *)(x) | *((unsigned char *)(x) + 1) << 8U |\
   *((unsigned char *)(x) + 2) << 16U | (opus_uint32)*((unsigned char *)(x) + 3) << 24U))

#define OP_CVTEPI8_EPI32_M32(x) \
 (_mm_cvtepi8_epi32(_mm_cvtsi32_si128(OP_LOADU_EPI32(x))))

#define OP_CVTEPI16_EPI32_M64(x) \
 (_mm_cvtepi16_epi32(_mm_loadl_epi64((__m128i *)(x))))

#endif
