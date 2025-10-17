/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 26, 2024.
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
#ifdef BN_MP_READ_SIGNED_BIN_C
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

/* read signed bin, big endian, first byte is 0==positive or 1==negative */
int mp_read_signed_bin (mp_int * a, const unsigned char *b, int c)
{
  int     res;

  /* read magnitude */
  if ((res = mp_read_unsigned_bin (a, b + 1, c - 1)) != MP_OKAY) {
    return res;
  }

  /* first byte is 0 for positive, non-zero for negative */
  if (b[0] == 0) {
     a->sign = MP_ZPOS;
  } else {
     a->sign = MP_NEG;
  }

  return MP_OKAY;
}
#endif

/* $Source: /cvs/libtom/libtommath/bn_mp_read_signed_bin.c,v $ */
/* $Revision: 1.5 $ */
/* $Date: 2006/12/28 01:25:13 $ */
