/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 20, 2023.
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

#include "macros.h"
#include "celt_lpc.h"
#include "stack_alloc.h"
#include "mathops.h"
#include "pitch.h"

#if defined(OPUS_X86_MAY_HAVE_SSE) && !defined(FIXED_POINT)

#include <xmmintrin.h>
#include "arch.h"

void xcorr_kernel_sse(const opus_val16 *x, const opus_val16 *y, opus_val32 sum[4], int len)
{
   int j;
   __m128 xsum1, xsum2;
   xsum1 = _mm_loadu_ps(sum);
   xsum2 = _mm_setzero_ps();

   for (j = 0; j < len-3; j += 4)
   {
      __m128 x0 = _mm_loadu_ps(x+j);
      __m128 yj = _mm_loadu_ps(y+j);
      __m128 y3 = _mm_loadu_ps(y+j+3);

      xsum1 = _mm_add_ps(xsum1,_mm_mul_ps(_mm_shuffle_ps(x0,x0,0x00),yj));
      xsum2 = _mm_add_ps(xsum2,_mm_mul_ps(_mm_shuffle_ps(x0,x0,0x55),
                                          _mm_shuffle_ps(yj,y3,0x49)));
      xsum1 = _mm_add_ps(xsum1,_mm_mul_ps(_mm_shuffle_ps(x0,x0,0xaa),
                                          _mm_shuffle_ps(yj,y3,0x9e)));
      xsum2 = _mm_add_ps(xsum2,_mm_mul_ps(_mm_shuffle_ps(x0,x0,0xff),y3));
   }
   if (j < len)
   {
      xsum1 = _mm_add_ps(xsum1,_mm_mul_ps(_mm_load1_ps(x+j),_mm_loadu_ps(y+j)));
      if (++j < len)
      {
         xsum2 = _mm_add_ps(xsum2,_mm_mul_ps(_mm_load1_ps(x+j),_mm_loadu_ps(y+j)));
         if (++j < len)
         {
            xsum1 = _mm_add_ps(xsum1,_mm_mul_ps(_mm_load1_ps(x+j),_mm_loadu_ps(y+j)));
         }
      }
   }
   _mm_storeu_ps(sum,_mm_add_ps(xsum1,xsum2));
}


void dual_inner_prod_sse(const opus_val16 *x, const opus_val16 *y01, const opus_val16 *y02,
      int N, opus_val32 *xy1, opus_val32 *xy2)
{
   int i;
   __m128 xsum1, xsum2;
   xsum1 = _mm_setzero_ps();
   xsum2 = _mm_setzero_ps();
   for (i=0;i<N-3;i+=4)
   {
      __m128 xi = _mm_loadu_ps(x+i);
      __m128 y1i = _mm_loadu_ps(y01+i);
      __m128 y2i = _mm_loadu_ps(y02+i);
      xsum1 = _mm_add_ps(xsum1,_mm_mul_ps(xi, y1i));
      xsum2 = _mm_add_ps(xsum2,_mm_mul_ps(xi, y2i));
   }
   /* Horizontal sum */
   xsum1 = _mm_add_ps(xsum1, _mm_movehl_ps(xsum1, xsum1));
   xsum1 = _mm_add_ss(xsum1, _mm_shuffle_ps(xsum1, xsum1, 0x55));
   _mm_store_ss(xy1, xsum1);
   xsum2 = _mm_add_ps(xsum2, _mm_movehl_ps(xsum2, xsum2));
   xsum2 = _mm_add_ss(xsum2, _mm_shuffle_ps(xsum2, xsum2, 0x55));
   _mm_store_ss(xy2, xsum2);
   for (;i<N;i++)
   {
      *xy1 = MAC16_16(*xy1, x[i], y01[i]);
      *xy2 = MAC16_16(*xy2, x[i], y02[i]);
   }
}

opus_val32 celt_inner_prod_sse(const opus_val16 *x, const opus_val16 *y,
      int N)
{
   int i;
   float xy;
   __m128 sum;
   sum = _mm_setzero_ps();
   /* FIXME: We should probably go 8-way and use 2 sums. */
   for (i=0;i<N-3;i+=4)
   {
      __m128 xi = _mm_loadu_ps(x+i);
      __m128 yi = _mm_loadu_ps(y+i);
      sum = _mm_add_ps(sum,_mm_mul_ps(xi, yi));
   }
   /* Horizontal sum */
   sum = _mm_add_ps(sum, _mm_movehl_ps(sum, sum));
   sum = _mm_add_ss(sum, _mm_shuffle_ps(sum, sum, 0x55));
   _mm_store_ss(&xy, sum);
   for (;i<N;i++)
   {
      xy = MAC16_16(xy, x[i], y[i]);
   }
   return xy;
}

void comb_filter_const_sse(opus_val32 *y, opus_val32 *x, int T, int N,
      opus_val16 g10, opus_val16 g11, opus_val16 g12)
{
   int i;
   __m128 x0v;
   __m128 g10v, g11v, g12v;
   g10v = _mm_load1_ps(&g10);
   g11v = _mm_load1_ps(&g11);
   g12v = _mm_load1_ps(&g12);
   x0v = _mm_loadu_ps(&x[-T-2]);
   for (i=0;i<N-3;i+=4)
   {
      __m128 yi, yi2, x1v, x2v, x3v, x4v;
      const opus_val32 *xp = &x[i-T-2];
      yi = _mm_loadu_ps(x+i);
      x4v = _mm_loadu_ps(xp+4);
#if 0
      /* Slower version with all loads */
      x1v = _mm_loadu_ps(xp+1);
      x2v = _mm_loadu_ps(xp+2);
      x3v = _mm_loadu_ps(xp+3);
#else
      x2v = _mm_shuffle_ps(x0v, x4v, 0x4e);
      x1v = _mm_shuffle_ps(x0v, x2v, 0x99);
      x3v = _mm_shuffle_ps(x2v, x4v, 0x99);
#endif

      yi = _mm_add_ps(yi, _mm_mul_ps(g10v,x2v));
#if 0 /* Set to 1 to make it bit-exact with the non-SSE version */
      yi = _mm_add_ps(yi, _mm_mul_ps(g11v,_mm_add_ps(x3v,x1v)));
      yi = _mm_add_ps(yi, _mm_mul_ps(g12v,_mm_add_ps(x4v,x0v)));
#else
      /* Use partial sums */
      yi2 = _mm_add_ps(_mm_mul_ps(g11v,_mm_add_ps(x3v,x1v)),
                       _mm_mul_ps(g12v,_mm_add_ps(x4v,x0v)));
      yi = _mm_add_ps(yi, yi2);
#endif
      x0v=x4v;
      _mm_storeu_ps(y+i, yi);
   }
#ifdef CUSTOM_MODES
   for (;i<N;i++)
   {
      y[i] = x[i]
               + MULT16_32_Q15(g10,x[i-T])
               + MULT16_32_Q15(g11,ADD32(x[i-T+1],x[i-T-1]))
               + MULT16_32_Q15(g12,ADD32(x[i-T+2],x[i-T-2]));
   }
#endif
}


#endif
