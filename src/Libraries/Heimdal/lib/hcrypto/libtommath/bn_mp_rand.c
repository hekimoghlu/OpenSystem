/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 5, 2024.
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
#ifdef BN_MP_RAND_C
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

/* makes a pseudo-random int of a given size */
int
mp_rand (mp_int * a, int digits)
{
  int     res;
  mp_digit d;

  mp_zero (a);
  if (digits <= 0) {
    return MP_OKAY;
  }

  /* first place a random non-zero digit */
  do {
    d = ((mp_digit) abs (rand ())) & MP_MASK;
  } while (d == 0);

  if ((res = mp_add_d (a, d, a)) != MP_OKAY) {
    return res;
  }

  while (--digits > 0) {
    if ((res = mp_lshd (a, 1)) != MP_OKAY) {
      return res;
    }

    if ((res = mp_add_d (a, ((mp_digit) abs (rand ())), a)) != MP_OKAY) {
      return res;
    }
  }

  return MP_OKAY;
}
#endif

/* $Source: /cvs/libtom/libtommath/bn_mp_rand.c,v $ */
/* $Revision: 1.4 $ */
/* $Date: 2006/12/28 01:25:13 $ */
