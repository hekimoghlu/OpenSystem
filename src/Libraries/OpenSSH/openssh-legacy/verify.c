/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 28, 2022.
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
/*
 * Public Domain, Author: Daniel J. Bernstein
 * Copied from nacl-20110221/crypto_verify/32/ref/verify.c
 */

#include "includes.h"

#include "crypto_api.h"

int crypto_verify_32(const unsigned char *x,const unsigned char *y)
{
  unsigned int differentbits = 0;
#define F(i) differentbits |= x[i] ^ y[i];
  F(0)
  F(1)
  F(2)
  F(3)
  F(4)
  F(5)
  F(6)
  F(7)
  F(8)
  F(9)
  F(10)
  F(11)
  F(12)
  F(13)
  F(14)
  F(15)
  F(16)
  F(17)
  F(18)
  F(19)
  F(20)
  F(21)
  F(22)
  F(23)
  F(24)
  F(25)
  F(26)
  F(27)
  F(28)
  F(29)
  F(30)
  F(31)
  return (1 & ((differentbits - 1) >> 8)) - 1;
}
