/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 27, 2025.
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

#include "macros.h"
#include "celt_lpc.h"
#include "stack_alloc.h"
#include "mathops.h"
#include "pitch.h"

#if defined(OPUS_X86_MAY_HAVE_SSE2) && defined(FIXED_POINT)
opus_val32 celt_inner_prod_sse2(const opus_val16 *x, const opus_val16 *y,
      int N)
{
    opus_int  i, dataSize16;
    opus_int32 sum;

    __m128i inVec1_76543210, inVec1_FEDCBA98, acc1;
    __m128i inVec2_76543210, inVec2_FEDCBA98, acc2;

    sum = 0;
    dataSize16 = N & ~15;

    acc1 = _mm_setzero_si128();
    acc2 = _mm_setzero_si128();

    for (i=0;i<dataSize16;i+=16)
    {
        inVec1_76543210 = _mm_loadu_si128((__m128i *)(&x[i + 0]));
        inVec2_76543210 = _mm_loadu_si128((__m128i *)(&y[i + 0]));

        inVec1_FEDCBA98 = _mm_loadu_si128((__m128i *)(&x[i + 8]));
        inVec2_FEDCBA98 = _mm_loadu_si128((__m128i *)(&y[i + 8]));

        inVec1_76543210 = _mm_madd_epi16(inVec1_76543210, inVec2_76543210);
        inVec1_FEDCBA98 = _mm_madd_epi16(inVec1_FEDCBA98, inVec2_FEDCBA98);

        acc1 = _mm_add_epi32(acc1, inVec1_76543210);
        acc2 = _mm_add_epi32(acc2, inVec1_FEDCBA98);
    }

    acc1 = _mm_add_epi32( acc1, acc2 );

    if (N - i >= 8)
    {
        inVec1_76543210 = _mm_loadu_si128((__m128i *)(&x[i + 0]));
        inVec2_76543210 = _mm_loadu_si128((__m128i *)(&y[i + 0]));

        inVec1_76543210 = _mm_madd_epi16(inVec1_76543210, inVec2_76543210);

        acc1 = _mm_add_epi32(acc1, inVec1_76543210);
        i += 8;
    }

    acc1 = _mm_add_epi32(acc1, _mm_unpackhi_epi64( acc1, acc1));
    acc1 = _mm_add_epi32(acc1, _mm_shufflelo_epi16( acc1, 0x0E));
    sum += _mm_cvtsi128_si32(acc1);

    for (;i<N;i++) {
        sum = silk_SMLABB(sum, x[i], y[i]);
    }

    return sum;
}
#endif
