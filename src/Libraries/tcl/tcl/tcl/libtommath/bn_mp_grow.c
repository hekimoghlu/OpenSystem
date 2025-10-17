/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 28, 2022.
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
#ifdef BN_MP_GROW_C
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

/* grow as required */
int mp_grow (mp_int * a, int size)
{
  int     i;
  mp_digit *tmp;

  /* if the alloc size is smaller alloc more ram */
  if (a->alloc < size) {
    /* ensure there are always at least MP_PREC digits extra on top */
    size += (MP_PREC * 2) - (size % MP_PREC);

    /* reallocate the array a->dp
     *
     * We store the return in a temporary variable
     * in case the operation failed we don't want
     * to overwrite the dp member of a.
     */
    tmp = OPT_CAST(mp_digit) XREALLOC (a->dp, sizeof (mp_digit) * size);
    if (tmp == NULL) {
      /* reallocation failed but "a" is still valid [can be freed] */
      return MP_MEM;
    }

    /* reallocation succeeded so set a->dp */
    a->dp = tmp;

    /* zero excess digits */
    i        = a->alloc;
    a->alloc = size;
    for (; i < a->alloc; i++) {
      a->dp[i] = 0;
    }
  }
  return MP_OKAY;
}
#endif

/* $Source: /cvsroot/tcl/libtommath/bn_mp_grow.c,v $ */
/* $Revision: 1.1.1.3 $ */
/* $Date: 2006/12/01 00:08:11 $ */
