/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 13, 2021.
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
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <xmmintrin.h>
#include <emmintrin.h>
#include <smmintrin.h>
#include "celt_lpc.h"
#include "stack_alloc.h"
#include "mathops.h"
#include "pitch.h"
#include "x86cpu.h"

#if defined(FIXED_POINT)

void celt_fir_sse4_1(const opus_val16 *x,
         const opus_val16 *num,
         opus_val16 *y,
         int N,
         int ord,
         int arch)
{
    int i,j;
    VARDECL(opus_val16, rnum);

    __m128i vecNoA;
    opus_int32 noA ;
    SAVE_STACK;

   ALLOC(rnum, ord, opus_val16);
   for(i=0;i<ord;i++)
      rnum[i] = num[ord-i-1];
   noA = EXTEND32(1) << SIG_SHIFT >> 1;
   vecNoA = _mm_set_epi32(noA, noA, noA, noA);

   for (i=0;i<N-3;i+=4)
   {
      opus_val32 sums[4] = {0};
      __m128i vecSum, vecX;

      xcorr_kernel(rnum, x+i-ord, sums, ord, arch);

      vecSum = _mm_loadu_si128((__m128i *)sums);
      vecSum = _mm_add_epi32(vecSum, vecNoA);
      vecSum = _mm_srai_epi32(vecSum, SIG_SHIFT);
      vecX = OP_CVTEPI16_EPI32_M64(x + i);
      vecSum = _mm_add_epi32(vecSum, vecX);
      vecSum = _mm_packs_epi32(vecSum, vecSum);
      _mm_storel_epi64((__m128i *)(y + i), vecSum);
   }
   for (;i<N;i++)
   {
      opus_val32 sum = 0;
      for (j=0;j<ord;j++)
         sum = MAC16_16(sum, rnum[j], x[i+j-ord]);
      y[i] = SATURATE16(ADD32(EXTEND32(x[i]), PSHR32(sum, SIG_SHIFT)));
   }

   RESTORE_STACK;
}

#endif
