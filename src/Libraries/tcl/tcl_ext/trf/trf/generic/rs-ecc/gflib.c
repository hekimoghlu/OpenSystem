/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 25, 2024.
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
/* gflib.c
	Math Library for GF[256]

	This file contains a number of mathematical functions for GF[256].
	Entry and result are always assumed to be in vector notation, since
	said notation allows for the zero element.  Attempting to reciprocate
	the zero element results in process exit 42.
*/

#include "gf.h"


/* Multiply two field elements */

unsigned char
gfmul (mul1, mul2)

     unsigned char mul1, mul2;
{
  unsigned char mul3;
  if (mul1 == 0 || mul2 == 0)
    mul3 = 0;
  else
    mul3 = e2v[(v2e[mul1] + v2e[mul2]) % 255];
  return (mul3);
}


/* Add two field elements.  Subtraction and addition are equivalent */

unsigned char
gfadd (add1, add2)

     unsigned char add1, add2;
{
  unsigned char add3;
  add3 = add1 ^ add2;
  return (add3);
}


/* Invert a field element, for division */

unsigned char
gfinv (ivt)

     unsigned char ivt;
{
  unsigned char ivtd;
  if (ivt == 0)
    exit (42);
  ivtd = e2v[255 - v2e[ivt]];
  return (ivtd);
}


/* Exponentiation.  Convert to exponential notation, mod 255 */

unsigned char
gfexp (mant, powr)

     unsigned char mant, powr;
{
  unsigned char expt;
  if (mant == 0)
    expt = 0;
  else
    expt = e2v[(v2e[mant] * powr) % 255];
  return (expt);
}
