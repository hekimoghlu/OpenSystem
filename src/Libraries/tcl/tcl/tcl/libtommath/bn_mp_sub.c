/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 25, 2023.
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
#ifdef BN_MP_SUB_C
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

/* high level subtraction (handles signs) */
int
mp_sub (mp_int * a, mp_int * b, mp_int * c)
{
  int     sa, sb, res;

  sa = a->sign;
  sb = b->sign;

  if (sa != sb) {
    /* subtract a negative from a positive, OR */
    /* subtract a positive from a negative. */
    /* In either case, ADD their magnitudes, */
    /* and use the sign of the first number. */
    c->sign = sa;
    res = s_mp_add (a, b, c);
  } else {
    /* subtract a positive from a positive, OR */
    /* subtract a negative from a negative. */
    /* First, take the difference between their */
    /* magnitudes, then... */
    if (mp_cmp_mag (a, b) != MP_LT) {
      /* Copy the sign from the first */
      c->sign = sa;
      /* The first has a larger or equal magnitude */
      res = s_mp_sub (a, b, c);
    } else {
      /* The result has the *opposite* sign from */
      /* the first number. */
      c->sign = (sa == MP_ZPOS) ? MP_NEG : MP_ZPOS;
      /* The second has a larger magnitude */
      res = s_mp_sub (b, a, c);
    }
  }
  return res;
}

#endif

/* $Source: /cvsroot/tcl/libtommath/bn_mp_sub.c,v $ */
/* $Revision: 1.1.1.3 $ */
/* $Date: 2006/12/01 00:08:11 $ */
