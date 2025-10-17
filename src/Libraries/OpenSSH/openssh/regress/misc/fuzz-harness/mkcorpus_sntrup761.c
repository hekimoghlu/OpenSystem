/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 5, 2023.
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

// Makes basic seed corpora for other fuzzers
//
// Will write to ./sntrup761_pubkey_corpus (for sntrup761_enc_fuzz) and
// to ./sntrup761_ciphertext_corpus (for sntrup761_dec_fuzz)

#include <sys/stat.h>
#include <stddef.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <err.h>
#include <errno.h>

#include "crypto_api.h"
#include "hash.c"

#undef randombytes
#define USE_SNTRUP761X25519 1
void randombytes(unsigned char *ptr, size_t l);
volatile crypto_int16 crypto_int16_optblocker = 0;
volatile crypto_int32 crypto_int32_optblocker = 0;
volatile crypto_int64 crypto_int64_optblocker = 0;
#include "sntrup761.c"

#define NSEEDS 1000

static int real_random;

void
randombytes(unsigned char *ptr, size_t l)
{
	if (real_random)
		arc4random_buf(ptr, l);
	else
		memset(ptr, 0, l);
}

void write_blob(const char *path, int n, const char *suffix,
    const void *ptr, size_t l)
{
	char name[256];
	FILE *f;

	snprintf(name, sizeof(name), "%s/%06d.%s", path, n, suffix);
	if ((f = fopen(name, "wb+")) == NULL)
		err(1, "fopen %s", name);
	if (fwrite(ptr, l, 1, f) != 1)
		err(1, "write %s", name);
	fclose(f);
}

int main(void)
{
	int i;
	unsigned char pk[crypto_kem_sntrup761_PUBLICKEYBYTES];
	unsigned char sk[crypto_kem_sntrup761_SECRETKEYBYTES];
	unsigned char ciphertext[crypto_kem_sntrup761_CIPHERTEXTBYTES];
	unsigned char secret[crypto_kem_sntrup761_BYTES];

	if (mkdir("sntrup761_pubkey_corpus", 0777) != 0 && errno != EEXIST)
		err(1, "mkdir sntrup761_pubkey_corpus");
	if (mkdir("sntrup761_ciphertext_corpus", 0777) != 0 && errno != EEXIST)
		err(1, "mkdir sntrup761_ciphertext_corpus");

	fprintf(stderr, "making: ");
	for (i = 0; i < NSEEDS; i++) {
		real_random = i != 0;
		if (crypto_kem_sntrup761_keypair(pk, sk) != 0)
			errx(1, "crypto_kem_sntrup761_keypair failed");
		write_blob("sntrup761_pubkey_corpus", i, "pk", pk, sizeof(pk));
		if (crypto_kem_sntrup761_enc(ciphertext, secret, pk) != 0)
			errx(1, "crypto_kem_sntrup761_enc failed");
		write_blob("sntrup761_ciphertext_corpus", i, "ct",
		    ciphertext, sizeof(ciphertext));
		if (i % 20 == 0)
			fprintf(stderr, ".");
	}
	fprintf(stderr, "\n");
	return 0;
}
