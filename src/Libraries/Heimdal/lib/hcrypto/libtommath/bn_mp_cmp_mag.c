/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 27, 2022.
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
#ifdef BN_MP_CMP_MAG_C
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

/* compare maginitude of two ints (unsigned) */
int mp_cmp_mag (mp_int * a, mp_int * b)
{
  int     n;
  mp_digit *tmpa, *tmpb;

  /* compare based on # of non-zero digits */
  if (a->used > b->used) {
    return MP_GT;
  }

  if (a->used < b->used) {
    return MP_LT;
  }

  /* alias for a */
  tmpa = a->dp + (a->used - 1);

  /* alias for b */
  tmpb = b->dp + (a->used - 1);

  /* compare based on digits  */
  for (n = 0; n < a->used; ++n, --tmpa, --tmpb) {
    if (*tmpa > *tmpb) {
      return MP_GT;
    }

    if (*tmpa < *tmpb) {
      return MP_LT;
    }
  }
  return MP_EQ;
}
#endif

/* $Source: /cvs/libtom/libtommath/bn_mp_cmp_mag.c,v $ */
/* $Revision: 1.4 $ */
/* $Date: 2006/12/28 01:25:13 $ */
