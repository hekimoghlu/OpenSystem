/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 4, 2022.
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
#ifdef BN_MP_FREAD_C
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

/* read a bigint from a file stream in ASCII */
int mp_fread(mp_int *a, int radix, FILE *stream)
{
   int err, ch, neg, y;

   /* clear a */
   mp_zero(a);

   /* if first digit is - then set negative */
   ch = fgetc(stream);
   if (ch == '-') {
      neg = MP_NEG;
      ch = fgetc(stream);
   } else {
      neg = MP_ZPOS;
   }

   for (;;) {
      /* find y in the radix map */
      for (y = 0; y < radix; y++) {
          if (mp_s_rmap[y] == ch) {
             break;
          }
      }
      if (y == radix) {
         break;
      }

      /* shift up and add */
      if ((err = mp_mul_d(a, radix, a)) != MP_OKAY) {
         return err;
      }
      if ((err = mp_add_d(a, y, a)) != MP_OKAY) {
         return err;
      }

      ch = fgetc(stream);
   }
   if (mp_cmp_d(a, 0) != MP_EQ) {
      a->sign = neg;
   }

   return MP_OKAY;
}

#endif

/* $Source: /cvs/libtom/libtommath/bn_mp_fread.c,v $ */
/* $Revision: 1.4 $ */
/* $Date: 2006/12/28 01:25:13 $ */
