/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 12, 2022.
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

// Basic fuzz test for encapsulate operation.

#include <stddef.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <err.h>

extern "C" {

#include "crypto_api.h"
#include "hash.c"

#undef randombytes
#define USE_SNTRUP761X25519 1
#ifdef SNTRUP761_NO_ASM
# undef __GNUC__
#endif
void randombytes(unsigned char *ptr, size_t l);
volatile crypto_int16 crypto_int16_optblocker = 0;
volatile crypto_int32 crypto_int32_optblocker = 0;
volatile crypto_int64 crypto_int64_optblocker = 0;
#include "sntrup761.c"

static int real_random;

void
randombytes(unsigned char *ptr, size_t l)
{
	if (real_random)
		arc4random_buf(ptr, l);
	else
		memset(ptr, 0, l);
}

int LLVMFuzzerTestOneInput(const uint8_t* input, size_t len)
{
	unsigned char pk[crypto_kem_sntrup761_PUBLICKEYBYTES];
	unsigned char ciphertext[crypto_kem_sntrup761_CIPHERTEXTBYTES];
	unsigned char secret[crypto_kem_sntrup761_BYTES];

	memset(&pk, 0, sizeof(pk));
	if (len > sizeof(pk)) {
		len = sizeof(pk);
	}
	memcpy(pk, input, len);

	real_random = 0;
	(void)crypto_kem_sntrup761_enc(ciphertext, secret, pk);
	real_random = 1;
	(void)crypto_kem_sntrup761_enc(ciphertext, secret, pk);
	return 0;
}

} // extern
