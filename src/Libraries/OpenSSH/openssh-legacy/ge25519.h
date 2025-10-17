/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 14, 2025.
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
 * Public Domain, Authors: Daniel J. Bernstein, Niels Duif, Tanja Lange,
 * Peter Schwabe, Bo-Yin Yang.
 * Copied from supercop-20130419/crypto_sign/ed25519/ref/ge25519.h
 */

#ifndef GE25519_H
#define GE25519_H

#include "fe25519.h"
#include "sc25519.h"

#define ge25519                           crypto_sign_ed25519_ref_ge25519
#define ge25519_base                      crypto_sign_ed25519_ref_ge25519_base
#define ge25519_unpackneg_vartime         crypto_sign_ed25519_ref_unpackneg_vartime
#define ge25519_pack                      crypto_sign_ed25519_ref_pack
#define ge25519_isneutral_vartime         crypto_sign_ed25519_ref_isneutral_vartime
#define ge25519_double_scalarmult_vartime crypto_sign_ed25519_ref_double_scalarmult_vartime
#define ge25519_scalarmult_base           crypto_sign_ed25519_ref_scalarmult_base

typedef struct
{
  fe25519 x;
  fe25519 y;
  fe25519 z;
  fe25519 t;
} ge25519;

extern const ge25519 ge25519_base;

int ge25519_unpackneg_vartime(ge25519 *r, const unsigned char p[32]);

void ge25519_pack(unsigned char r[32], const ge25519 *p);

int ge25519_isneutral_vartime(const ge25519 *p);

void ge25519_double_scalarmult_vartime(ge25519 *r, const ge25519 *p1, const sc25519 *s1, const ge25519 *p2, const sc25519 *s2);

void ge25519_scalarmult_base(ge25519 *r, const sc25519 *s);

#endif
