/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 20, 2024.
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

#ifndef _CORECRYPTO_CCEC25519_H_
#define _CORECRYPTO_CCEC25519_H_

#include <corecrypto/cc.h>
#include <corecrypto/ccrng.h>
#include <corecrypto/ccdigest.h>

typedef uint8_t ccec25519key[32];
typedef ccec25519key ccec25519secretkey;
typedef ccec25519key ccec25519pubkey;
typedef ccec25519key ccec25519base;

typedef uint8_t ccec25519signature[64];

// these parameters are all pretty much just uint8_t arrays, but since the types are there, we might as well use them Â¯\_(ãƒ„)_/Â¯
//
// shared_secret: OUT parameter
// private_key: IN parameter
// public_key: IN parameter
int cccurve25519(ccec25519key shared_secret, ccec25519secretkey private_key, ccec25519pubkey public_key);

// rng: REGULAR parameter
// public_key: OUT parameter
// private_key: OUT parameter
int cccurve25519_make_key_pair(struct ccrng_state* rng, ccec25519pubkey public_key, ccec25519secretkey private_key);

#endif

