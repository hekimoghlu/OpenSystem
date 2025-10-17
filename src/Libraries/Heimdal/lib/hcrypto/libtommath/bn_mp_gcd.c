/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 18, 2024.
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
#ifdef BN_MP_GCD_C
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

/* Greatest Common Divisor using the binary method */
int mp_gcd (mp_int * a, mp_int * b, mp_int * c)
{
  mp_int  u, v;
  int     k, u_lsb, v_lsb, res;

  /* either zero than gcd is the largest */
  if (mp_iszero (a) == MP_YES) {
    return mp_abs (b, c);
  }
  if (mp_iszero (b) == MP_YES) {
    return mp_abs (a, c);
  }

  /* get copies of a and b we can modify */
  if ((res = mp_init_copy (&u, a)) != MP_OKAY) {
    return res;
  }

  if ((res = mp_init_copy (&v, b)) != MP_OKAY) {
    goto LBL_U;
  }

  /* must be positive for the remainder of the algorithm */
  u.sign = v.sign = MP_ZPOS;

  /* B1.  Find the common power of two for u and v */
  u_lsb = mp_cnt_lsb(&u);
  v_lsb = mp_cnt_lsb(&v);
  k     = MIN(u_lsb, v_lsb);

  if (k > 0) {
     /* divide the power of two out */
     if ((res = mp_div_2d(&u, k, &u, NULL)) != MP_OKAY) {
        goto LBL_V;
     }

     if ((res = mp_div_2d(&v, k, &v, NULL)) != MP_OKAY) {
        goto LBL_V;
     }
  }

  /* divide any remaining factors of two out */
  if (u_lsb != k) {
     if ((res = mp_div_2d(&u, u_lsb - k, &u, NULL)) != MP_OKAY) {
        goto LBL_V;
     }
  }

  if (v_lsb != k) {
     if ((res = mp_div_2d(&v, v_lsb - k, &v, NULL)) != MP_OKAY) {
        goto LBL_V;
     }
  }

  while (mp_iszero(&v) == 0) {
     /* make sure v is the largest */
     if (mp_cmp_mag(&u, &v) == MP_GT) {
        /* swap u and v to make sure v is >= u */
        mp_exch(&u, &v);
     }

     /* subtract smallest from largest */
     if ((res = s_mp_sub(&v, &u, &v)) != MP_OKAY) {
        goto LBL_V;
     }

     /* Divide out all factors of two */
     if ((res = mp_div_2d(&v, mp_cnt_lsb(&v), &v, NULL)) != MP_OKAY) {
        goto LBL_V;
     }
  }

  /* multiply by 2**k which we divided out at the beginning */
  if ((res = mp_mul_2d (&u, k, c)) != MP_OKAY) {
     goto LBL_V;
  }
  c->sign = MP_ZPOS;
  res = MP_OKAY;
LBL_V:mp_clear (&u);
LBL_U:mp_clear (&v);
  return res;
}
#endif

/* $Source: /cvs/libtom/libtommath/bn_mp_gcd.c,v $ */
/* $Revision: 1.5 $ */
/* $Date: 2006/12/28 01:25:13 $ */
