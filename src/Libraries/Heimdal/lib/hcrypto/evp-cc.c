/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 12, 2024.
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
/* CommonCrypto provider */

#ifdef __APPLE__

#include "config.h"

#include <sys/types.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#ifdef HAVE_COMMONCRYPTO_COMMONDIGEST_H
#include <CommonCrypto/CommonDigest.h>
#endif
#ifdef HAVE_COMMONCRYPTO_COMMONCRYPTOR_H
#include <CommonCrypto/CommonCryptor.h>
#endif

#include <evp.h>
#include <evp-cc.h>

/*
 *
 */

#ifdef HAVE_COMMONCRYPTO_COMMONCRYPTOR_H

struct cc_key {
    CCCryptorRef href;
};

static int
cc_do_cipher(EVP_CIPHER_CTX *ctx,
	     unsigned char *out,
	     const unsigned char *in,
	     unsigned int size)
{
    struct cc_key *cc = ctx->cipher_data;
    CCCryptorStatus ret;
    size_t moved;

    memcpy(out, in, size);

    ret = CCCryptorUpdate(cc->href, in, size, out, size, &moved);
    if (ret)
	return 0;

    if (moved != size)
	return 0;

    return 1;
}

static int
cc_do_cfb8_cipher(EVP_CIPHER_CTX *ctx,
                  unsigned char *out,
                  const unsigned char *in,
                  unsigned int size)
{
    struct cc_key *cc = ctx->cipher_data;
    CCCryptorStatus ret;
    size_t moved;
    unsigned int i;

    for (i = 0; i < size; i++) {
        unsigned char oiv[EVP_MAX_IV_LENGTH + 1];

        assert(ctx->cipher->iv_len + 1 <= sizeof(oiv));
        memcpy(oiv, ctx->iv, ctx->cipher->iv_len);

        ret = CCCryptorUpdate(cc->href, ctx->iv, ctx->cipher->iv_len,
                              ctx->iv, ctx->cipher->iv_len, &moved);
        if (ret)
            return 0;

        if (moved != ctx->cipher->iv_len)
            return 0;

        if (!ctx->encrypt)
            oiv[ctx->cipher->iv_len] = in[i];
        out[i] = in[i] ^ ctx->iv[0];
        if (ctx->encrypt)
            oiv[ctx->cipher->iv_len] = out[i];

        memcpy(ctx->iv, &oiv[1], ctx->cipher->iv_len);
    }

    return 1;
}

static int
cc_cleanup(EVP_CIPHER_CTX *ctx)
{
    struct cc_key *cc = ctx->cipher_data;
    if (cc->href)
	CCCryptorRelease(cc->href);
    return 1;
}

static int
init_cc_key(int encp, CCAlgorithm alg, CCOptions opts, const void *key,
	    size_t keylen, const void *iv, CCCryptorRef *ref)
{
    CCOperation op = encp ? kCCEncrypt : kCCDecrypt;
    CCCryptorStatus ret;

    if (*ref) {
	if (key == NULL && iv) {
	    CCCryptorReset(*ref, iv);
	    return 1;
	}
	CCCryptorRelease(*ref);
    }

    ret = CCCryptorCreate(op, alg, opts, key, keylen, iv, ref);
    if (ret)
	return 0;
    return 1;
}

static int
cc_des_ede3_cbc_init(EVP_CIPHER_CTX *ctx,
		     const unsigned char * key,
		     const unsigned char * iv,
		     int encp)
{
    struct cc_key *cc = ctx->cipher_data;
    return init_cc_key(encp, kCCAlgorithm3DES, 0, key, kCCKeySize3DES, iv, &cc->href);
}

#endif /* HAVE_COMMONCRYPTO_COMMONCRYPTOR_H */

/**
 * The tripple DES cipher type (Apple CommonCrypto provider)
 *
 * @return the DES-EDE3-CBC EVP_CIPHER pointer.
 *
 * @ingroup hcrypto_evp
 */

const EVP_CIPHER *
EVP_cc_des_ede3_cbc(void)
{
#ifdef HAVE_COMMONCRYPTO_COMMONCRYPTOR_H
    static const EVP_CIPHER des_ede3_cbc = {
	0,
	8,
	24,
	8,
	EVP_CIPH_CBC_MODE|EVP_CIPH_ALWAYS_CALL_INIT,
	cc_des_ede3_cbc_init,
	cc_do_cipher,
	cc_cleanup,
	sizeof(struct cc_key),
	NULL,
	NULL,
	NULL,
	NULL
    };
    return &des_ede3_cbc;
#else
    return NULL;
#endif
}

#ifdef HAVE_COMMONCRYPTO_COMMONCRYPTOR_H
/*
 *
 */

static int
cc_des_cbc_init(EVP_CIPHER_CTX *ctx,
		const unsigned char * key,
		const unsigned char * iv,
		int encp)
{
    struct cc_key *cc = ctx->cipher_data;
    return init_cc_key(encp, kCCAlgorithmDES, 0, key, kCCBlockSizeDES, iv, &cc->href);
}
#endif

/**
 * The DES cipher type (Apple CommonCrypto provider)
 *
 * @return the DES-CBC EVP_CIPHER pointer.
 *
 * @ingroup hcrypto_evp
 */

const EVP_CIPHER *
EVP_cc_des_cbc(void)
{
#ifdef HAVE_COMMONCRYPTO_COMMONCRYPTOR_H
    static const EVP_CIPHER des_ede3_cbc = {
	0,
	kCCBlockSizeDES,
	kCCBlockSizeDES,
	kCCBlockSizeDES,
	EVP_CIPH_CBC_MODE|EVP_CIPH_ALWAYS_CALL_INIT,
	cc_des_cbc_init,
	cc_do_cipher,
	cc_cleanup,
	sizeof(struct cc_key),
	NULL,
	NULL,
	NULL,
	NULL
    };
    return &des_ede3_cbc;
#else
    return NULL;
#endif
}

#ifdef HAVE_COMMONCRYPTO_COMMONCRYPTOR_H
/*
 *
 */

static int
cc_aes_cbc_init(EVP_CIPHER_CTX *ctx,
		const unsigned char * key,
		const unsigned char * iv,
		int encp)
{
    struct cc_key *cc = ctx->cipher_data;
    return init_cc_key(encp, kCCAlgorithmAES128, 0, key, ctx->cipher->key_len, iv, &cc->href);
}
#endif

/**
 * The AES-128 cipher type (Apple CommonCrypto provider)
 *
 * @return the AES-128-CBC EVP_CIPHER pointer.
 *
 * @ingroup hcrypto_evp
 */

const EVP_CIPHER *
EVP_cc_aes_128_cbc(void)
{
#ifdef HAVE_COMMONCRYPTO_COMMONCRYPTOR_H
    static const EVP_CIPHER c = {
	0,
	kCCBlockSizeAES128,
	kCCKeySizeAES128,
	kCCBlockSizeAES128,
	EVP_CIPH_CBC_MODE|EVP_CIPH_ALWAYS_CALL_INIT,
	cc_aes_cbc_init,
	cc_do_cipher,
	cc_cleanup,
	sizeof(struct cc_key),
	NULL,
	NULL,
	NULL,
	NULL
    };
    return &c;
#else
    return NULL;
#endif
}

/**
 * The AES-192 cipher type (Apple CommonCrypto provider)
 *
 * @return the AES-192-CBC EVP_CIPHER pointer.
 *
 * @ingroup hcrypto_evp
 */

const EVP_CIPHER *
EVP_cc_aes_192_cbc(void)
{
#ifdef HAVE_COMMONCRYPTO_COMMONCRYPTOR_H
    static const EVP_CIPHER c = {
	0,
	kCCBlockSizeAES128,
	kCCKeySizeAES192,
	kCCBlockSizeAES128,
	EVP_CIPH_CBC_MODE|EVP_CIPH_ALWAYS_CALL_INIT,
	cc_aes_cbc_init,
	cc_do_cipher,
	cc_cleanup,
	sizeof(struct cc_key),
	NULL,
	NULL,
	NULL,
	NULL
    };
    return &c;
#else
    return NULL;
#endif
}

/**
 * The AES-256 cipher type (Apple CommonCrypto provider)
 *
 * @return the AES-256-CBC EVP_CIPHER pointer.
 *
 * @ingroup hcrypto_evp
 */

const EVP_CIPHER *
EVP_cc_aes_256_cbc(void)
{
#ifdef HAVE_COMMONCRYPTO_COMMONCRYPTOR_H
    static const EVP_CIPHER c = {
	0,
	kCCBlockSizeAES128,
	kCCKeySizeAES256,
	kCCBlockSizeAES128,
	EVP_CIPH_CBC_MODE|EVP_CIPH_ALWAYS_CALL_INIT,
	cc_aes_cbc_init,
	cc_do_cipher,
	cc_cleanup,
	sizeof(struct cc_key),
	NULL,
	NULL,
	NULL,
	NULL
    };
    return &c;
#else
    return NULL;
#endif
}

#ifdef HAVE_COMMONCRYPTO_COMMONCRYPTOR_H
/*
 *
 */

static int
cc_aes_cfb8_init(EVP_CIPHER_CTX *ctx,
		const unsigned char * key,
		const unsigned char * iv,
		int encp)
{
    struct cc_key *cc = ctx->cipher_data;
    memcpy(ctx->iv, iv, ctx->cipher->iv_len);
    return init_cc_key(1, kCCAlgorithmAES128, kCCOptionECBMode,
		       key, ctx->cipher->key_len, NULL, &cc->href);
}
#endif

/**
 * The AES-128 CFB8 cipher type (Apple CommonCrypto provider)
 *
 * @return the AES-128-CFB8 EVP_CIPHER pointer.
 *
 * @ingroup hcrypto_evp
 */

const EVP_CIPHER *
EVP_cc_aes_128_cfb8(void)
{
#ifdef HAVE_COMMONCRYPTO_COMMONCRYPTOR_H
    static const EVP_CIPHER c = {
	0,
	1,
	kCCKeySizeAES128,
	kCCBlockSizeAES128,
	EVP_CIPH_CFB8_MODE|EVP_CIPH_ALWAYS_CALL_INIT,
	cc_aes_cfb8_init,
	cc_do_cfb8_cipher,
	cc_cleanup,
	sizeof(struct cc_key),
	NULL,
	NULL,
	NULL,
	NULL
    };
    return &c;
#else
    return NULL;
#endif
}

/**
 * The AES-192 CFB8 cipher type (Apple CommonCrypto provider)
 *
 * @return the AES-192-CFB8 EVP_CIPHER pointer.
 *
 * @ingroup hcrypto_evp
 */

const EVP_CIPHER *
EVP_cc_aes_192_cfb8(void)
{
#ifdef HAVE_COMMONCRYPTO_COMMONCRYPTOR_H
    static const EVP_CIPHER c = {
	0,
	1,
	kCCKeySizeAES192,
	kCCBlockSizeAES128,
	EVP_CIPH_CFB8_MODE|EVP_CIPH_ALWAYS_CALL_INIT,
	cc_aes_cfb8_init,
	cc_do_cfb8_cipher,
	cc_cleanup,
	sizeof(struct cc_key),
	NULL,
	NULL,
	NULL,
	NULL
    };
    return &c;
#else
    return NULL;
#endif
}

/**
 * The AES-256 CFB8 cipher type (Apple CommonCrypto provider)
 *
 * @return the AES-256-CFB8 EVP_CIPHER pointer.
 *
 * @ingroup hcrypto_evp
 */

const EVP_CIPHER *
EVP_cc_aes_256_cfb8(void)
{
#ifdef HAVE_COMMONCRYPTO_COMMONCRYPTOR_H
    static const EVP_CIPHER c = {
	0,
	kCCBlockSizeAES128,
	kCCKeySizeAES256,
	kCCBlockSizeAES128,
	EVP_CIPH_CFB8_MODE|EVP_CIPH_ALWAYS_CALL_INIT,
	cc_aes_cfb8_init,
	cc_do_cfb8_cipher,
	cc_cleanup,
	sizeof(struct cc_key),
	NULL,
	NULL,
	NULL,
	NULL
    };
    return &c;
#else
    return NULL;
#endif
}

/*
 *
 */

#ifdef COMMONCRYPTO_SUPPORTS_RC2
static int
cc_rc2_cbc_init(EVP_CIPHER_CTX *ctx,
		const unsigned char * key,
		const unsigned char * iv,
		int encp)
{
    struct cc_key *cc = ctx->cipher_data;
    return init_cc_key(encp, kCCAlgorithmRC2, 0, key, ctx->cipher->key_len, iv, &cc->href);
}
#endif

/**
 * The RC2 cipher type - common crypto
 *
 * @return the RC2 EVP_CIPHER pointer.
 *
 * @ingroup hcrypto_evp
 */


const EVP_CIPHER *
EVP_cc_rc2_cbc(void)
{
#ifdef COMMONCRYPTO_SUPPORTS_RC2
    static const EVP_CIPHER rc2_cbc = {
	0,
	kCCBlockSizeRC2,
	16,
	kCCBlockSizeRC2,
	EVP_CIPH_CBC_MODE|EVP_CIPH_ALWAYS_CALL_INIT,
	cc_rc2_cbc_init,
	cc_do_cipher,
	cc_cleanup,
	sizeof(struct cc_key),
	NULL,
	NULL,
	NULL,
	NULL
    };
    return &rc2_cbc;
#else
    return NULL;
#endif
}

/**
 * The RC2-40 cipher type - common crypto
 *
 * @return the RC2-40 EVP_CIPHER pointer.
 *
 * @ingroup hcrypto_evp
 */


const EVP_CIPHER *
EVP_cc_rc2_40_cbc(void)
{
#ifdef COMMONCRYPTO_SUPPORTS_RC2
    static const EVP_CIPHER rc2_40_cbc = {
	0,
	kCCBlockSizeRC2,
	5,
	kCCBlockSizeRC2,
	EVP_CIPH_CBC_MODE|EVP_CIPH_ALWAYS_CALL_INIT,
	cc_rc2_cbc_init,
	cc_do_cipher,
	cc_cleanup,
	sizeof(struct cc_key),
	NULL,
	NULL,
	NULL,
	NULL
    };
    return &rc2_40_cbc;
#else
    return NULL;
#endif
}


/**
 * The RC2-64 cipher type - common crypto
 *
 * @return the RC2-64 EVP_CIPHER pointer.
 *
 * @ingroup hcrypto_evp
 */


const EVP_CIPHER *
EVP_cc_rc2_64_cbc(void)
{
#ifdef COMMONCRYPTO_SUPPORTS_RC2
    static const EVP_CIPHER rc2_64_cbc = {
	0,
	kCCBlockSizeRC2,
	8,
	kCCBlockSizeRC2,
	EVP_CIPH_CBC_MODE|EVP_CIPH_ALWAYS_CALL_INIT,
	cc_rc2_cbc_init,
	cc_do_cipher,
	cc_cleanup,
	sizeof(struct cc_key),
	NULL,
	NULL,
	NULL,
	NULL
    };
    return &rc2_64_cbc;
#else
    return NULL;
#endif
}

/**
 * The CommonCrypto md4 provider
 *
 * @ingroup hcrypto_evp
 */

const EVP_MD *
EVP_cc_md4(void)
{
#ifdef HAVE_COMMONCRYPTO_COMMONDIGEST_H
    static const struct hc_evp_md md4 = {
	CC_MD4_DIGEST_LENGTH,
	CC_MD4_BLOCK_BYTES,
	sizeof(CC_MD4_CTX),
	(hc_evp_md_init)CC_MD4_Init,
	(hc_evp_md_update)CC_MD4_Update,
	(hc_evp_md_final)CC_MD4_Final,
	(hc_evp_md_cleanup)NULL
    };
    return &md4;
#else
    return NULL;
#endif
}

/**
 * The CommonCrypto md5 provider
 *
 * @ingroup hcrypto_evp
 */

const EVP_MD *
EVP_cc_md5(void)
{
#ifdef HAVE_COMMONCRYPTO_COMMONDIGEST_H
    static const struct hc_evp_md md5 = {
	CC_MD5_DIGEST_LENGTH,
	CC_MD5_BLOCK_BYTES,
	sizeof(CC_MD5_CTX),
	(hc_evp_md_init)CC_MD5_Init,
	(hc_evp_md_update)CC_MD5_Update,
	(hc_evp_md_final)CC_MD5_Final,
	(hc_evp_md_cleanup)NULL
    };
    return &md5;
#else
    return NULL;
#endif
}

/**
 * The CommonCrypto sha1 provider
 *
 * @ingroup hcrypto_evp
 */

const EVP_MD *
EVP_cc_sha1(void)
{
#ifdef HAVE_COMMONCRYPTO_COMMONDIGEST_H
    static const struct hc_evp_md sha1 = {
	CC_SHA1_DIGEST_LENGTH,
	CC_SHA1_BLOCK_BYTES,
	sizeof(CC_SHA1_CTX),
	(hc_evp_md_init)CC_SHA1_Init,
	(hc_evp_md_update)CC_SHA1_Update,
	(hc_evp_md_final)CC_SHA1_Final,
	(hc_evp_md_cleanup)NULL
    };
    return &sha1;
#else
    return NULL;
#endif
}

/**
 * The CommonCrypto sha256 provider
 *
 * @ingroup hcrypto_evp
 */

const EVP_MD *
EVP_cc_sha256(void)
{
#ifdef HAVE_COMMONCRYPTO_COMMONDIGEST_H
    static const struct hc_evp_md sha256 = {
	CC_SHA256_DIGEST_LENGTH,
	CC_SHA256_BLOCK_BYTES,
	sizeof(CC_SHA256_CTX),
	(hc_evp_md_init)CC_SHA256_Init,
	(hc_evp_md_update)CC_SHA256_Update,
	(hc_evp_md_final)CC_SHA256_Final,
	(hc_evp_md_cleanup)NULL
    };
    return &sha256;
#else
    return NULL;
#endif
}

/**
 * The CommonCrypto sha384 provider
 *
 * @ingroup hcrypto_evp
 */

const EVP_MD *
EVP_cc_sha384(void)
{
#ifdef HAVE_COMMONCRYPTO_COMMONDIGEST_H
    static const struct hc_evp_md sha384 = {
	CC_SHA384_DIGEST_LENGTH,
	CC_SHA384_BLOCK_BYTES,
	sizeof(CC_SHA512_CTX),
	(hc_evp_md_init)CC_SHA384_Init,
	(hc_evp_md_update)CC_SHA384_Update,
	(hc_evp_md_final)CC_SHA384_Final,
	(hc_evp_md_cleanup)NULL
    };
    return &sha384;
#else
    return NULL;
#endif
}

/**
 * The CommonCrypto sha512 provider
 *
 * @ingroup hcrypto_evp
 */

const EVP_MD *
EVP_cc_sha512(void)
{
#ifdef HAVE_COMMONCRYPTO_COMMONDIGEST_H
    static const struct hc_evp_md sha512 = {
	CC_SHA512_DIGEST_LENGTH,
	CC_SHA512_BLOCK_BYTES,
	sizeof(CC_SHA512_CTX),
	(hc_evp_md_init)CC_SHA512_Init,
	(hc_evp_md_update)CC_SHA512_Update,
	(hc_evp_md_final)CC_SHA512_Final,
	(hc_evp_md_cleanup)NULL
    };
    return &sha512;
#else
    return NULL;
#endif
}

/**
 * The Camellia-128 cipher type - CommonCrypto
 *
 * @return the Camellia-128 EVP_CIPHER pointer.
 *
 * @ingroup hcrypto_evp
 */

const EVP_CIPHER *
EVP_cc_camellia_128_cbc(void)
{
    return NULL;
}

/**
 * The Camellia-198 cipher type - CommonCrypto
 *
 * @return the Camellia-198 EVP_CIPHER pointer.
 *
 * @ingroup hcrypto_evp
 */

const EVP_CIPHER *
EVP_cc_camellia_192_cbc(void)
{
    return NULL;
}

/**
 * The Camellia-256 cipher type - CommonCrypto
 *
 * @return the Camellia-256 EVP_CIPHER pointer.
 *
 * @ingroup hcrypto_evp
 */

const EVP_CIPHER *
EVP_cc_camellia_256_cbc(void)
{
    return NULL;
}

#ifdef HAVE_COMMONCRYPTO_COMMONCRYPTOR_H

/*
 *
 */

static int
cc_rc4_init(EVP_CIPHER_CTX *ctx,
	    const unsigned char * key,
	    const unsigned char * iv,
	    int encp)
{
    struct cc_key *cc = ctx->cipher_data;
    return init_cc_key(encp, kCCAlgorithmRC4, 0, key, ctx->key_len, iv, &cc->href);
}

#endif

/**

 * The RC4 cipher type (Apple CommonCrypto provider)
 *
 * @return the RC4 EVP_CIPHER pointer.
 *
 * @ingroup hcrypto_evp
 */

const EVP_CIPHER *
EVP_cc_rc4(void)
{
#ifdef HAVE_COMMONCRYPTO_COMMONCRYPTOR_H
    static const EVP_CIPHER rc4 = {
	0,
	1,
	16,
	0,
	EVP_CIPH_STREAM_CIPHER|EVP_CIPH_VARIABLE_LENGTH,
	cc_rc4_init,
	cc_do_cipher,
	cc_cleanup,
	sizeof(struct cc_key),
	NULL,
	NULL,
	NULL,
	NULL
    };
    return &rc4;
#else
    return NULL;
#endif
}


/**
 * The RC4-40 cipher type (Apple CommonCrypto provider)
 *
 * @return the RC4 EVP_CIPHER pointer.
 *
 * @ingroup hcrypto_evp
 */

const EVP_CIPHER *
EVP_cc_rc4_40(void)
{
#ifdef HAVE_COMMONCRYPTO_COMMONCRYPTOR_H
    static const EVP_CIPHER rc4_40 = {
	0,
	1,
	5,
	0,
	EVP_CIPH_STREAM_CIPHER|EVP_CIPH_VARIABLE_LENGTH,
	cc_rc4_init,
	cc_do_cipher,
	cc_cleanup,
	sizeof(struct cc_key),
	NULL,
	NULL,
	NULL,
	NULL
    };
    return &rc4_40;
#else
    return NULL;
#endif
}

#endif /* __APPLE__ */

