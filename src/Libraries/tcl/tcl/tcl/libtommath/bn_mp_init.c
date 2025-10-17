/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 28, 2022.
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
#ifdef BN_MP_INIT_C
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

/* init a new mp_int */
int mp_init (mp_int * a)
{
  int i;

  /* allocate memory required and clear it */
  a->dp = OPT_CAST(mp_digit) XMALLOC (sizeof (mp_digit) * MP_PREC);
  if (a->dp == NULL) {
    return MP_MEM;
  }

  /* set the digits to zero */
  for (i = 0; i < MP_PREC; i++) {
      a->dp[i] = 0;
  }

  /* set the used to zero, allocated digits to the default precision
   * and sign to positive */
  a->used  = 0;
  a->alloc = MP_PREC;
  a->sign  = MP_ZPOS;

  return MP_OKAY;
}
#endif

/* $Source: /cvsroot/tcl/libtommath/bn_mp_init.c,v $ */
/* $Revision: 1.1.1.3 $ */
/* $Date: 2006/12/01 00:08:11 $ */
