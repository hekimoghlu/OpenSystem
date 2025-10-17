/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 13, 2024.
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
#include "krb5_locl.h"

void
_krb5_evp_schedule(krb5_context context,
		   struct _krb5_key_type *kt,
		   struct _krb5_key_data *kd)
{
    struct _krb5_evp_schedule *key = kd->schedule->data;
    const EVP_CIPHER *c = (*kt->evp)();

    EVP_CIPHER_CTX_init(&key->ectx);
    EVP_CIPHER_CTX_init(&key->dctx);

    EVP_CipherInit_ex(&key->ectx, c, NULL, kd->key->keyvalue.data, NULL, 1);
    EVP_CipherInit_ex(&key->dctx, c, NULL, kd->key->keyvalue.data, NULL, 0);
}

void
_krb5_evp_cleanup(krb5_context context, struct _krb5_key_data *kd)
{
    struct _krb5_evp_schedule *key = kd->schedule->data;
    EVP_CIPHER_CTX_cleanup(&key->ectx);
    EVP_CIPHER_CTX_cleanup(&key->dctx);
}

krb5_error_code
_krb5_evp_encrypt(krb5_context context,
		struct _krb5_key_data *key,
		void *data,
		size_t len,
		krb5_boolean encryptp,
		int usage,
		void *ivec)
{
    struct _krb5_evp_schedule *ctx = key->schedule->data;
    EVP_CIPHER_CTX *c;
    c = encryptp ? &ctx->ectx : &ctx->dctx;
    if (ivec == NULL) {
	/* alloca ? */
	size_t len2 = EVP_CIPHER_CTX_iv_length(c);
	void *loiv = malloc(len2);
	if (loiv == NULL) {
	    krb5_clear_error_message(context);
	    return ENOMEM;
	}
	memset(loiv, 0, len2);
	EVP_CipherInit_ex(c, NULL, NULL, NULL, loiv, -1);
	free(loiv);
    } else
	EVP_CipherInit_ex(c, NULL, NULL, NULL, ivec, -1);
    EVP_Cipher(c, data, data, len);
    return 0;
}

static const unsigned char zero_ivec[EVP_MAX_BLOCK_LENGTH] = { 0 };

krb5_error_code
_krb5_evp_encrypt_cts(krb5_context context,
		      struct _krb5_key_data *key,
		      void *data,
		      size_t len,
		      krb5_boolean encryptp,
		      int usage,
		      void *ivec)
{
    size_t i, blocksize;
    struct _krb5_evp_schedule *ctx = key->schedule->data;
    unsigned char tmp[EVP_MAX_BLOCK_LENGTH], ivec2[EVP_MAX_BLOCK_LENGTH];
    EVP_CIPHER_CTX *c;
    unsigned char *p;

    c = encryptp ? &ctx->ectx : &ctx->dctx;

    blocksize = EVP_CIPHER_CTX_block_size(c);

    if (len < blocksize) {
	krb5_set_error_message(context, EINVAL,
			       "message block too short");
	return EINVAL;
    } else if (len == blocksize) {
	EVP_CipherInit_ex(c, NULL, NULL, NULL, zero_ivec, -1);
	EVP_Cipher(c, data, data, len);
	return 0;
    }

    if (ivec)
	EVP_CipherInit_ex(c, NULL, NULL, NULL, ivec, -1);
    else
	EVP_CipherInit_ex(c, NULL, NULL, NULL, zero_ivec, -1);

    if (encryptp) {

	p = data;
	i = ((len - 1) / blocksize) * blocksize;
	EVP_Cipher(c, p, p, i);
	p += i - blocksize;
	len -= i;
	memcpy(ivec2, p, blocksize);

	for (i = 0; i < len; i++)
	    tmp[i] = p[i + blocksize] ^ ivec2[i];
	for (; i < blocksize; i++)
	    tmp[i] = 0 ^ ivec2[i];

	EVP_CipherInit_ex(c, NULL, NULL, NULL, zero_ivec, -1);
	EVP_Cipher(c, p, tmp, blocksize);

	memcpy(p + blocksize, ivec2, len);
	if (ivec)
	    memcpy(ivec, p, blocksize);
    } else {
	unsigned char tmp2[EVP_MAX_BLOCK_LENGTH], tmp3[EVP_MAX_BLOCK_LENGTH];

	p = data;
	if (len > blocksize * 2) {
	    /* remove last two blocks and round up, decrypt this with cbc, then do cts dance */
	    i = ((((len - blocksize * 2) + blocksize - 1) / blocksize) * blocksize);
	    memcpy(ivec2, p + i - blocksize, blocksize);
	    EVP_Cipher(c, p, p, i);
	    p += i;
	    len -= i + blocksize;
	} else {
	    if (ivec)
		memcpy(ivec2, ivec, blocksize);
	    else
		memcpy(ivec2, zero_ivec, blocksize);
	    len -= blocksize;
	}

	memcpy(tmp, p, blocksize);
	EVP_CipherInit_ex(c, NULL, NULL, NULL, zero_ivec, -1);
	EVP_Cipher(c, tmp2, p, blocksize);

	memcpy(tmp3, p + blocksize, len);
	memcpy(tmp3 + len, tmp2 + len, blocksize - len); /* xor 0 */

	for (i = 0; i < len; i++)
	    p[i + blocksize] = tmp2[i] ^ tmp3[i];

	EVP_CipherInit_ex(c, NULL, NULL, NULL, zero_ivec, -1);
	EVP_Cipher(c, p, tmp3, blocksize);

	for (i = 0; i < blocksize; i++)
	    p[i] ^= ivec2[i];
	if (ivec)
	    memcpy(ivec, tmp, blocksize);
    }
    return 0;
}
