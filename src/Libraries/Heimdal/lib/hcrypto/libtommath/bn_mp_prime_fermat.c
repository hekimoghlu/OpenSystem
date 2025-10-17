/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 9, 2024.
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
#ifdef BN_MP_PRIME_FERMAT_C
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

/* performs one Fermat test.
 *
 * If "a" were prime then b**a == b (mod a) since the order of
 * the multiplicative sub-group would be phi(a) = a-1.  That means
 * it would be the same as b**(a mod (a-1)) == b**1 == b (mod a).
 *
 * Sets result to 1 if the congruence holds, or zero otherwise.
 */
int mp_prime_fermat (mp_int * a, mp_int * b, int *result)
{
  mp_int  t;
  int     err;

  /* default to composite  */
  *result = MP_NO;

  /* ensure b > 1 */
  if (mp_cmp_d(b, 1) != MP_GT) {
     return MP_VAL;
  }

  /* init t */
  if ((err = mp_init (&t)) != MP_OKAY) {
    return err;
  }

  /* compute t = b**a mod a */
  if ((err = mp_exptmod (b, a, a, &t)) != MP_OKAY) {
    goto LBL_T;
  }

  /* is it equal to b? */
  if (mp_cmp (&t, b) == MP_EQ) {
    *result = MP_YES;
  }

  err = MP_OKAY;
LBL_T:mp_clear (&t);
  return err;
}
#endif

/* $Source: /cvs/libtom/libtommath/bn_mp_prime_fermat.c,v $ */
/* $Revision: 1.4 $ */
/* $Date: 2006/12/28 01:25:13 $ */
