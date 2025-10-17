/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 7, 2022.
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

#include <tommath.h>
#ifdef BN_MP_REDUCE_IS_2K_C
/* LibTomMath, multiple-precision integer library -- Tom St Denis
 *
 * LibTomMath is a library that provides multiple-precision
 * integer arithmetic as well as number theoretic functionality.
 *
 * The library was designed directly after the MPI library by
 * Michael Fromberger but has been written from scratch with
 * additional optimizations in place.
 *
 * The library is free for all purposes without any express
 * guarantee it works.
 *
 * Tom St Denis, tomstdenis@gmail.com, http://libtom.org
 */

/* determines if mp_reduce_2k can be used */
int mp_reduce_is_2k(mp_int *a)
{
   int ix, iy, iw;
   mp_digit iz;

   if (a->used == 0) {
      return MP_NO;
   } else if (a->used == 1) {
      return MP_YES;
   } else if (a->used > 1) {
      iy = mp_count_bits(a);
      iz = 1;
      iw = 1;

      /* Test every bit from the second digit up, must be 1 */
      for (ix = DIGIT_BIT; ix < iy; ix++) {
          if ((a->dp[iw] & iz) == 0) {
             return MP_NO;
          }
          iz <<= 1;
          if (iz > (mp_digit)MP_MASK) {
             ++iw;
             iz = 1;
          }
      }
   }
   return MP_YES;
}

#endif

/* $Source: /cvs/libtom/libtommath/bn_mp_reduce_is_2k.c,v $ */
/* $Revision: 1.4 $ */
/* $Date: 2006/12/28 01:25:13 $ */
