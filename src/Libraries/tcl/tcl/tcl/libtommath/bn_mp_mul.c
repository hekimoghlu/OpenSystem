/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 30, 2023.
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
#ifdef BN_MP_MUL_C
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
 * Tom St Denis, tomstdenis@gmail.com, http://math.libtomcrypt.com
 */

/* high level multiplication (handles sign) */
int mp_mul (mp_int * a, mp_int * b, mp_int * c)
{
  int     res, neg;
  neg = (a->sign == b->sign) ? MP_ZPOS : MP_NEG;

  /* use Toom-Cook? */
#ifdef BN_MP_TOOM_MUL_C
  if (MIN (a->used, b->used) >= TOOM_MUL_CUTOFF) {
    res = mp_toom_mul(a, b, c);
  } else 
#endif
#ifdef BN_MP_KARATSUBA_MUL_C
  /* use Karatsuba? */
  if (MIN (a->used, b->used) >= KARATSUBA_MUL_CUTOFF) {
    res = mp_karatsuba_mul (a, b, c);
  } else 
#endif
  {
    /* can we use the fast multiplier?
     *
     * The fast multiplier can be used if the output will 
     * have less than MP_WARRAY digits and the number of 
     * digits won't affect carry propagation
     */
    int     digs = a->used + b->used + 1;

#ifdef BN_FAST_S_MP_MUL_DIGS_C
    if ((digs < MP_WARRAY) &&
        MIN(a->used, b->used) <= 
        (1 << ((CHAR_BIT * sizeof (mp_word)) - (2 * DIGIT_BIT)))) {
      res = fast_s_mp_mul_digs (a, b, c, digs);
    } else 
#endif
#ifdef BN_S_MP_MUL_DIGS_C
      res = s_mp_mul (a, b, c); /* uses s_mp_mul_digs */
#else
      res = MP_VAL;
#endif

  }
  c->sign = (c->used > 0) ? neg : MP_ZPOS;
  return res;
}
#endif

/* $Source: /cvsroot/tcl/libtommath/bn_mp_mul.c,v $ */
/* $Revision: 1.1.1.3 $ */
/* $Date: 2006/12/01 00:08:11 $ */
