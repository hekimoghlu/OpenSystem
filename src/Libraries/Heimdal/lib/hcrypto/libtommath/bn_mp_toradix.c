/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 15, 2024.
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
#ifdef BN_MP_TORADIX_C
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

/* stores a bignum as a ASCII string in a given radix (2..64) */
int mp_toradix (mp_int * a, char *str, int radix)
{
  int     res, digs;
  mp_int  t;
  mp_digit d;
  char   *_s = str;

  /* check range of the radix */
  if (radix < 2 || radix > 64) {
    return MP_VAL;
  }

  /* quick out if its zero */
  if (mp_iszero(a) == 1) {
     *str++ = '0';
     *str = '\0';
     return MP_OKAY;
  }

  if ((res = mp_init_copy (&t, a)) != MP_OKAY) {
    return res;
  }

  /* if it is negative output a - */
  if (t.sign == MP_NEG) {
    ++_s;
    *str++ = '-';
    t.sign = MP_ZPOS;
  }

  digs = 0;
  while (mp_iszero (&t) == 0) {
    if ((res = mp_div_d (&t, (mp_digit) radix, &t, &d)) != MP_OKAY) {
      mp_clear (&t);
      return res;
    }
    *str++ = mp_s_rmap[d];
    ++digs;
  }

  /* reverse the digits of the string.  In this case _s points
   * to the first digit [exluding the sign] of the number]
   */
  bn_reverse ((unsigned char *)_s, digs);

  /* append a NULL so the string is properly terminated */
  *str = '\0';

  mp_clear (&t);
  return MP_OKAY;
}

#endif

/* $Source: /cvs/libtom/libtommath/bn_mp_toradix.c,v $ */
/* $Revision: 1.4 $ */
/* $Date: 2006/12/28 01:25:13 $ */
