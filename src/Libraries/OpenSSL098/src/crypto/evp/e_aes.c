/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 10, 2021.
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
#include <openssl/opensslconf.h>
#ifndef OPENSSL_NO_AES
#include <openssl/evp.h>
#include <openssl/err.h>
#include <string.h>
#include <assert.h>
#include <openssl/aes.h>
#include "evp_locl.h"

static int aes_init_key(EVP_CIPHER_CTX *ctx, const unsigned char *key,
					const unsigned char *iv, int enc);

typedef struct
	{
	AES_KEY ks;
	} EVP_AES_KEY;

#define data(ctx)	EVP_C_DATA(EVP_AES_KEY,ctx)

IMPLEMENT_BLOCK_CIPHER(aes_128, ks, AES, EVP_AES_KEY,
		       NID_aes_128, 16, 16, 16, 128,
		       EVP_CIPH_FLAG_FIPS|EVP_CIPH_FLAG_DEFAULT_ASN1,
		       aes_init_key,
		       NULL, NULL, NULL, NULL)
IMPLEMENT_BLOCK_CIPHER(aes_192, ks, AES, EVP_AES_KEY,
		       NID_aes_192, 16, 24, 16, 128,
		       EVP_CIPH_FLAG_FIPS|EVP_CIPH_FLAG_DEFAULT_ASN1,
		       aes_init_key,
		       NULL, NULL, NULL, NULL)
IMPLEMENT_BLOCK_CIPHER(aes_256, ks, AES, EVP_AES_KEY,
		       NID_aes_256, 16, 32, 16, 128,
		       EVP_CIPH_FLAG_FIPS|EVP_CIPH_FLAG_DEFAULT_ASN1,
		       aes_init_key,
		       NULL, NULL, NULL, NULL)

#define IMPLEMENT_AES_CFBR(ksize,cbits,flags)	IMPLEMENT_CFBR(aes,AES,EVP_AES_KEY,ks,ksize,cbits,16,flags)

IMPLEMENT_AES_CFBR(128,1,EVP_CIPH_FLAG_FIPS)
IMPLEMENT_AES_CFBR(192,1,EVP_CIPH_FLAG_FIPS)
IMPLEMENT_AES_CFBR(256,1,EVP_CIPH_FLAG_FIPS)

IMPLEMENT_AES_CFBR(128,8,EVP_CIPH_FLAG_FIPS)
IMPLEMENT_AES_CFBR(192,8,EVP_CIPH_FLAG_FIPS)
IMPLEMENT_AES_CFBR(256,8,EVP_CIPH_FLAG_FIPS)

static int aes_init_key(EVP_CIPHER_CTX *ctx, const unsigned char *key,
		   const unsigned char *iv, int enc)
	{
	int ret;

	if ((ctx->cipher->flags & EVP_CIPH_MODE) == EVP_CIPH_CFB_MODE
	    || (ctx->cipher->flags & EVP_CIPH_MODE) == EVP_CIPH_OFB_MODE
	    || enc) 
		ret=AES_set_encrypt_key(key, ctx->key_len * 8, ctx->cipher_data);
	else
		ret=AES_set_decrypt_key(key, ctx->key_len * 8, ctx->cipher_data);

	if(ret < 0)
		{
		EVPerr(EVP_F_AES_INIT_KEY,EVP_R_AES_KEY_SETUP_FAILED);
		return 0;
		}

	return 1;
	}

#endif
