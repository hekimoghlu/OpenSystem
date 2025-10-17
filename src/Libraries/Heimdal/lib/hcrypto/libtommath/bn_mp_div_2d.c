/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 8, 2024.
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
#ifdef BN_MP_DIV_2D_C
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

/* shift right by a certain bit count (store quotient in c, optional remainder in d) */
int mp_div_2d (mp_int * a, int b, mp_int * c, mp_int * d)
{
  mp_digit D, r, rr;
  int     x, res;
  mp_int  t;


  /* if the shift count is <= 0 then we do no work */
  if (b <= 0) {
    res = mp_copy (a, c);
    if (d != NULL) {
      mp_zero (d);
    }
    return res;
  }

  if ((res = mp_init (&t)) != MP_OKAY) {
    return res;
  }

  /* get the remainder */
  if (d != NULL) {
    if ((res = mp_mod_2d (a, b, &t)) != MP_OKAY) {
      mp_clear (&t);
      return res;
    }
  }

  /* copy */
  if ((res = mp_copy (a, c)) != MP_OKAY) {
    mp_clear (&t);
    return res;
  }

  /* shift by as many digits in the bit count */
  if (b >= (int)DIGIT_BIT) {
    mp_rshd (c, b / DIGIT_BIT);
  }

  /* shift any bit count < DIGIT_BIT */
  D = (mp_digit) (b % DIGIT_BIT);
  if (D != 0) {
    register mp_digit *tmpc, mask, shift;

    /* mask */
    mask = (((mp_digit)1) << D) - 1;

    /* shift for lsb */
    shift = DIGIT_BIT - D;

    /* alias */
    tmpc = c->dp + (c->used - 1);

    /* carry */
    r = 0;
    for (x = c->used - 1; x >= 0; x--) {
      /* get the lower  bits of this word in a temp */
      rr = *tmpc & mask;

      /* shift the current word and mix in the carry bits from the previous word */
      *tmpc = (*tmpc >> D) | (r << shift);
      --tmpc;

      /* set the carry to the carry bits of the current word found above */
      r = rr;
    }
  }
  mp_clamp (c);
  if (d != NULL) {
    mp_exch (&t, d);
  }
  mp_clear (&t);
  return MP_OKAY;
}
#endif

/* $Source: /cvs/libtom/libtommath/bn_mp_div_2d.c,v $ */
/* $Revision: 1.4 $ */
/* $Date: 2006/12/28 01:25:13 $ */
