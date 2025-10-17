/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 8, 2023.
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
#ifdef BN_MP_REDUCE_2K_L_C
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

/* reduces a modulo n where n is of the form 2**p - d
   This differs from reduce_2k since "d" can be larger
   than a single digit.
*/
int mp_reduce_2k_l(mp_int *a, mp_int *n, mp_int *d)
{
   mp_int q;
   int    p, res;

   if ((res = mp_init(&q)) != MP_OKAY) {
      return res;
   }

   p = mp_count_bits(n);
top:
   /* q = a/2**p, a = a mod 2**p */
   if ((res = mp_div_2d(a, p, &q, a)) != MP_OKAY) {
      goto ERR;
   }

   /* q = q * d */
   if ((res = mp_mul(&q, d, &q)) != MP_OKAY) {
      goto ERR;
   }

   /* a = a + q */
   if ((res = s_mp_add(a, &q, a)) != MP_OKAY) {
      goto ERR;
   }

   if (mp_cmp_mag(a, n) != MP_LT) {
      s_mp_sub(a, n, a);
      goto top;
   }

ERR:
   mp_clear(&q);
   return res;
}

#endif

/* $Source: /cvs/libtom/libtommath/bn_mp_reduce_2k_l.c,v $ */
/* $Revision: 1.4 $ */
/* $Date: 2006/12/28 01:25:13 $ */
