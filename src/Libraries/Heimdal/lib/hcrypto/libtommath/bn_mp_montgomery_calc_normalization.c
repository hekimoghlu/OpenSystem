/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 30, 2025.
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
#ifdef BN_MP_MONTGOMERY_CALC_NORMALIZATION_C
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

/*
 * shifts with subtractions when the result is greater than b.
 *
 * The method is slightly modified to shift B unconditionally upto just under
 * the leading bit of b.  This saves alot of multiple precision shifting.
 */
int mp_montgomery_calc_normalization (mp_int * a, mp_int * b)
{
  int     x, bits, res;

  /* how many bits of last digit does b use */
  bits = mp_count_bits (b) % DIGIT_BIT;

  if (b->used > 1) {
     if ((res = mp_2expt (a, (b->used - 1) * DIGIT_BIT + bits - 1)) != MP_OKAY) {
        return res;
     }
  } else {
     mp_set(a, 1);
     bits = 1;
  }


  /* now compute C = A * B mod b */
  for (x = bits - 1; x < (int)DIGIT_BIT; x++) {
    if ((res = mp_mul_2 (a, a)) != MP_OKAY) {
      return res;
    }
    if (mp_cmp_mag (a, b) != MP_LT) {
      if ((res = s_mp_sub (a, b, a)) != MP_OKAY) {
        return res;
      }
    }
  }

  return MP_OKAY;
}
#endif

/* $Source: /cvs/libtom/libtommath/bn_mp_montgomery_calc_normalization.c,v $ */
/* $Revision: 1.4 $ */
/* $Date: 2006/12/28 01:25:13 $ */
