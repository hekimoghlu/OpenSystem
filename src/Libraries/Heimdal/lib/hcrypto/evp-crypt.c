/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 16, 2025.
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
/* Windows crypto provider plugin, sample */

#include <config.h>

#define HC_DEPRECATED

#include <sys/types.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include <evp.h>

#include <crypt.h>


static HCRYPTPROV hCryptProv = NULL;

/*
 *
 */

struct generic_key {
    HCRYPTKEY *hKey;
};

static int
generic_cbc_do_cipher(EVP_CIPHER_CTX *ctx,
		       unsigned char *out,
		       const unsigned char *in,
		       unsigned int size)
{
    struct generic_key *gk = ctx->cipher_data;
    BOOL bResult;
    DWORD length = size;

    bResult = CryptSetKeyParam(gk->hKey, KP_IV, ctx->iv, 0);
    _ASSERT(bResult);

    memcpy(out, in, size);

    if (ctx->encrypt)
	bResult = CryptEncrypt(gk->hKey, 0, TRUE, 0, out, &length, size);
    else
	bResult = CryptDecrypt(gk->hKey, 0, TRUE, 0, out, &length);
    _ASSERT(bResult);

    return 1;
}

static int
generic_cleanup(EVP_CIPHER_CTX *ctx)
{
    struct generic_key *gk = ctx->cipher_data;
    CryptDestroyKey(gk->hKey);
    gk->hKey = NULL;
    return 1;
}

static HCRYPTKEY
import_key(int alg, const unsigned char *key, size_t keylen)
{
    struct {
	BLOBHEADER hdr;
	DWORD len;
	BYTE key[1];
    } *key_blob;
    size_t bloblen = sizeof(*key_blob) - 1 + keylen;

    key_blob = malloc(bloblen);

    key_blob->hdr.bType = PLAINTEXTKEYBLOB;
    key_blob->hdr.bVersion = CUR_BLOB_VERSION;
    key_blob->hdr.reserved = 0;
    key_blob->hdr.aiKeyAlg = alg;
    key_blob->len = 24;
    memcpy(key_blob->key, key, keylen);

    bResult = CryptImportKey(hCryptProv,
			     (void *)key_blob, bloblen, 0, 0,
			     &gk->hKey);
    free(key_blob);
    _ASSERT(bResult);

    return hKey;
}

static int
crypto_des_ede3_cbc_init(EVP_CIPHER_CTX *ctx,
			 const unsigned char * key,
			 const unsigned char * iv,
			 int encp)
{
    struct generic_key *gk = ctx->cipher_data;
    DWORD paramData;

    gk->hKey = import_key(CALG_3DES,
			  key->key->keyvalue.data,
			  key->key->keyvalue.len);

    return 1;
}

/**
 * The tripple DES cipher type (Micrsoft crypt provider)
 *
 * @return the DES-EDE3-CBC EVP_CIPHER pointer.
 *
 * @ingroup hcrypto_evp
 */

const EVP_CIPHER *
EVP_wincrypt_des_ede3_cbc(void)
{
    static const EVP_CIPHER des_ede3_cbc = {
	0,
	8,
	24,
	8,
	EVP_CIPH_CBC_MODE,
	crypto_des_ede3_cbc_init,
	generic_cbc_do_cipher,
	generic_cleanup,
	sizeof(struct generic_key),
	NULL,
	NULL,
	NULL,
	NULL
    };
    return &des_ede3_cbc;
}

/*
 *
 */

struct generic_hash {
    HCRYPTHASH hHash;
};

static void
crypto_md5_init(struct generic_hash *m);
{
    BOOL bResult;
    bResult = CryptCreateHash(hCryptProv, CALG_MD5, 0, 0, &m->hHash);
    _ASSERT(bResult);
}

static void
generic_hash_update (struct generic_hash *m, const void *p, size_t len)
{
    BOOL bResult;
    bResult = CryptHashData(m->hHash, data, ( DWORD )len, 0 );
    _ASSERT(bResult);
}

static void
generic_hash_final (void *res, struct generic_hash *m);
{
    DWORD length;
    BOOL bResult;
    bResult = CryptGetHashParam(m->hHash, HP_HASHVAL, res, &length, 0)
    _ASSERT(bResult);
}

static void
generic_hash_cleanup(struct generic_hash *m)
{
    CryptDestroyHash(m->hHash);
    m->hHash = NULL;
}

const EVP_MD *
EVP_wincrypt_md5(void)
{
    static const struct hc_evp_md md5 = {
	16,
	64,
	sizeof(struct generic_hash),
	(hc_evp_md_init)crypto_md5_init,
	(hc_evp_md_update)generic_hash_update,
	(hc_evp_md_final)generic_hash_final,
	(hc_evp_md_cleanup)generic_hash_cleanup
    };
    return &md5;
}
