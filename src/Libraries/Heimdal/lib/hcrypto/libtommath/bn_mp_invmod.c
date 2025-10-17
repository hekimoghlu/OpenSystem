/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 24, 2024.
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
#ifdef BN_MP_INVMOD_C
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

/* hac 14.61, pp608 */
int mp_invmod (mp_int * a, mp_int * b, mp_int * c)
{
  /* b cannot be negative */
  if (b->sign == MP_NEG || mp_iszero(b) == 1) {
    return MP_VAL;
  }

#ifdef BN_FAST_MP_INVMOD_C
  /* if the modulus is odd we can use a faster routine instead */
  if (mp_isodd (b) == 1) {
    return fast_mp_invmod (a, b, c);
  }
#endif

#ifdef BN_MP_INVMOD_SLOW_C
  return mp_invmod_slow(a, b, c);
#else
  return MP_VAL;
#endif
}
#endif

/* $Source: /cvs/libtom/libtommath/bn_mp_invmod.c,v $ */
/* $Revision: 1.4 $ */
/* $Date: 2006/12/28 01:25:13 $ */
