/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 1, 2022.
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

#include "rijndael-alg-fst.h"
#include "aes.h"

int
AES_set_encrypt_key(const unsigned char *userkey, const int bits, AES_KEY *key)
{
    key->rounds = rijndaelKeySetupEnc(key->key, userkey, bits);
    if (key->rounds == 0)
	return -1;
    return 0;
}

int
AES_set_decrypt_key(const unsigned char *userkey, const int bits, AES_KEY *key)
{
    key->rounds = rijndaelKeySetupDec(key->key, userkey, bits);
    if (key->rounds == 0)
	return -1;
    return 0;
}

void
AES_encrypt(const unsigned char *in, unsigned char *out, const AES_KEY *key)
{
    rijndaelEncrypt(key->key, key->rounds, in, out);
}

void
AES_decrypt(const unsigned char *in, unsigned char *out, const AES_KEY *key)
{
    rijndaelDecrypt(key->key, key->rounds, in, out);
}

void
AES_cbc_encrypt(const unsigned char *in, unsigned char *out,
		unsigned long size, const AES_KEY *key,
		unsigned char *iv, int forward_encrypt)
{
    unsigned char tmp[AES_BLOCK_SIZE];
    int i;

    if (forward_encrypt) {
	while (size >= AES_BLOCK_SIZE) {
	    for (i = 0; i < AES_BLOCK_SIZE; i++)
		tmp[i] = in[i] ^ iv[i];
	    AES_encrypt(tmp, out, key);
	    memcpy(iv, out, AES_BLOCK_SIZE);
	    size -= AES_BLOCK_SIZE;
	    in += AES_BLOCK_SIZE;
	    out += AES_BLOCK_SIZE;
	}
	if (size) {
	    for (i = 0; i < size; i++)
		tmp[i] = in[i] ^ iv[i];
	    for (i = size; i < AES_BLOCK_SIZE; i++)
		tmp[i] = iv[i];
	    AES_encrypt(tmp, out, key);
	    memcpy(iv, out, AES_BLOCK_SIZE);
	}
    } else {
	while (size >= AES_BLOCK_SIZE) {
	    memcpy(tmp, in, AES_BLOCK_SIZE);
	    AES_decrypt(tmp, out, key);
	    for (i = 0; i < AES_BLOCK_SIZE; i++)
		out[i] ^= iv[i];
	    memcpy(iv, tmp, AES_BLOCK_SIZE);
	    size -= AES_BLOCK_SIZE;
	    in += AES_BLOCK_SIZE;
	    out += AES_BLOCK_SIZE;
	}
	if (size) {
	    memcpy(tmp, in, AES_BLOCK_SIZE);
	    AES_decrypt(tmp, out, key);
	    for (i = 0; i < size; i++)
		out[i] ^= iv[i];
	    memcpy(iv, tmp, AES_BLOCK_SIZE);
	}
    }
}

void
AES_cfb8_encrypt(const unsigned char *in, unsigned char *out,
                 unsigned long size, const AES_KEY *key,
                 unsigned char *iv, int forward_encrypt)
{
    int i;

    for (i = 0; i < size; i++) {
        unsigned char tmp[AES_BLOCK_SIZE + 1];

        memcpy(tmp, iv, AES_BLOCK_SIZE);
        AES_encrypt(iv, iv, key);
        if (!forward_encrypt) {
            tmp[AES_BLOCK_SIZE] = in[i];
        }
        out[i] = in[i] ^ iv[0];
        if (forward_encrypt) {
            tmp[AES_BLOCK_SIZE] = out[i];
        }
        memcpy(iv, &tmp[1], AES_BLOCK_SIZE);
    }
}
