/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 27, 2022.
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
#include "config.h"

#ifdef KRB5
#include <krb5-types.h>
#endif

#include <string.h>

#include "camellia-ntt.h"
#include "camellia.h"

#include <roken.h>

int
CAMELLIA_set_key(const unsigned char *userkey,
		 const int bits, CAMELLIA_KEY *key)
{
    key->bits = bits;
    Camellia_Ekeygen(bits, userkey, key->key);
    return 1;
}

void
CAMELLIA_encrypt(const unsigned char *in, unsigned char *out,
		 const CAMELLIA_KEY *key)
{
    Camellia_EncryptBlock(key->bits, in, key->key, out);

}

void
CAMELLIA_decrypt(const unsigned char *in, unsigned char *out,
		 const CAMELLIA_KEY *key)
{
    Camellia_DecryptBlock(key->bits, in, key->key, out);
}

void
CAMELLIA_cbc_encrypt(const unsigned char *in, unsigned char *out,
		     unsigned long size, const CAMELLIA_KEY *key,
		     unsigned char *iv, int mode_encrypt)
{
    unsigned char tmp[CAMELLIA_BLOCK_SIZE];
    int i;

    if (mode_encrypt) {
	while (size >= CAMELLIA_BLOCK_SIZE) {
	    for (i = 0; i < CAMELLIA_BLOCK_SIZE; i++)
		tmp[i] = in[i] ^ iv[i];
	    CAMELLIA_encrypt(tmp, out, key);
	    memcpy(iv, out, CAMELLIA_BLOCK_SIZE);
	    size -= CAMELLIA_BLOCK_SIZE;
	    in += CAMELLIA_BLOCK_SIZE;
	    out += CAMELLIA_BLOCK_SIZE;
	}
	if (size) {
	    for (i = 0; i < size; i++)
		tmp[i] = in[i] ^ iv[i];
	    for (i = size; i < CAMELLIA_BLOCK_SIZE; i++)
		tmp[i] = iv[i];
	    CAMELLIA_encrypt(tmp, out, key);
	    memcpy(iv, out, CAMELLIA_BLOCK_SIZE);
	}
    } else {
	while (size >= CAMELLIA_BLOCK_SIZE) {
	    memcpy(tmp, in, CAMELLIA_BLOCK_SIZE);
	    CAMELLIA_decrypt(tmp, out, key);
	    for (i = 0; i < CAMELLIA_BLOCK_SIZE; i++)
		out[i] ^= iv[i];
	    memcpy(iv, tmp, CAMELLIA_BLOCK_SIZE);
	    size -= CAMELLIA_BLOCK_SIZE;
	    in += CAMELLIA_BLOCK_SIZE;
	    out += CAMELLIA_BLOCK_SIZE;
	}
	if (size) {
	    memcpy(tmp, in, CAMELLIA_BLOCK_SIZE);
	    CAMELLIA_decrypt(tmp, out, key);
	    for (i = 0; i < size; i++)
		out[i] ^= iv[i];
	    memcpy(iv, tmp, CAMELLIA_BLOCK_SIZE);
	}
    }
}
