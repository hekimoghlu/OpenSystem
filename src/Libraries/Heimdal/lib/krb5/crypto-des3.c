/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 24, 2024.
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

#ifdef HEIM_KRB5_DES3

/*
 *
 */

static void
DES3_random_key(krb5_context context,
		krb5_keyblock *key)
{
    uint8_t *k = key->keyvalue.data;
    do {
	krb5_generate_random_block(key->keyvalue.data, key->keyvalue.length);
	CCDesSetOddParity(&k[0], 8);
	CCDesSetOddParity(&k[8], 8);
	CCDesSetOddParity(&k[16], 8);
    } while(CCDesIsWeakKey(&k[0], 8) ||
	    CCDesIsWeakKey(&k[8], 8) ||
	    CCDesIsWeakKey(&k[16], 8));
}

static krb5_error_code
DES3_prf(krb5_context context,
	 krb5_crypto crypto,
	 const krb5_data *in,
	 krb5_data *out)
{
    struct _krb5_checksum_type *ct = crypto->et->checksum;
    krb5_error_code ret;
    Checksum result;
    krb5_keyblock *derived;
    size_t prfsize;

    result.cksumtype = ct->type;
    ret = krb5_data_alloc(&result.checksum, ct->checksumsize);
    if (ret) {
	krb5_set_error_message(context, ret, N_("malloc: out memory", ""));
	return ret;
    }

    ret = (*ct->checksum)(context, NULL, in->data, in->length, 0, &result);
    if (ret) {
	krb5_data_free(&result.checksum);
	return ret;
    }

    if (result.checksum.length < crypto->et->blocksize)
	krb5_abortx(context, "internal prf error");

    derived = NULL;
    ret = krb5_derive_key(context, crypto->key.key,
			  crypto->et->type, "prf", 3, &derived);
    if (ret)
	krb5_abortx(context, "krb5_derive_key");

    prfsize = (result.checksum.length / crypto->et->blocksize) * crypto->et->blocksize;
    heim_assert(prfsize == crypto->et->prf_length, "prfsize not same ?");

    ret = krb5_data_alloc(out, prfsize);
    if (ret)
	krb5_abortx(context, "malloc failed");

    {
	const EVP_CIPHER *c = (*crypto->et->keytype->evp)();
	EVP_CIPHER_CTX ctx;

	EVP_CIPHER_CTX_init(&ctx); /* ivec all zero */
	EVP_CipherInit_ex(&ctx, c, NULL, derived->keyvalue.data, NULL, 1);
	EVP_Cipher(&ctx, out->data, result.checksum.data, prfsize);
	EVP_CIPHER_CTX_cleanup(&ctx);
    }

    krb5_data_free(&result.checksum);
    krb5_free_keyblock(context, derived);

    return ret;
}


#ifdef DES3_OLD_ENCTYPE
static struct _krb5_key_type keytype_des3 = {
    ETYPE_OLD_DES3_CBC_SHA1,
    "des3",
    168,
    24,
    sizeof(struct _krb5_evp_schedule),
    DES3_random_key,
    _krb5_evp_schedule,
    _krb5_des3_salt,
    _krb5_DES3_random_to_key,
    _krb5_evp_cleanup,
    EVP_des_ede3_cbc
};
#endif

static struct _krb5_key_type keytype_des3_derived = {
    ETYPE_OLD_DES3_CBC_SHA1,
    "des3",
    168,
    24,
    sizeof(struct _krb5_evp_schedule),
    DES3_random_key,
    _krb5_evp_schedule,
    _krb5_des3_salt_derived,
    _krb5_DES3_random_to_key,
    _krb5_evp_cleanup,
    EVP_des_ede3_cbc
};

#ifdef DES3_OLD_ENCTYPE
static krb5_error_code
RSA_MD5_DES3_checksum(krb5_context context,
		      struct _krb5_key_data *key,
		      const void *data,
		      size_t len,
		      unsigned usage,
		      Checksum *C)
{
    return _krb5_des_checksum(context, kCCDigestMD5, key, data, len, C);
}

static krb5_error_code
RSA_MD5_DES3_verify(krb5_context context,
		    struct _krb5_key_data *key,
		    const void *data,
		    size_t len,
		    unsigned usage,
		    Checksum *C)
{
    return _krb5_des_verify(context, kCCDigestMD5, key, data, len, C);
}

struct _krb5_checksum_type _krb5_checksum_rsa_md5_des3 = {
    CKSUMTYPE_RSA_MD5_DES3,
    "rsa-md5-des3",
    64,
    24,
    F_KEYED | F_CPROOF | F_VARIANT,
    RSA_MD5_DES3_checksum,
    RSA_MD5_DES3_verify
};
#endif

struct _krb5_checksum_type _krb5_checksum_hmac_sha1_des3 = {
    CKSUMTYPE_HMAC_SHA1_DES3,
    "hmac-sha1-des3",
    64,
    20,
    F_KEYED | F_CPROOF | F_DERIVED | F_WARNING,
    _krb5_SP_HMAC_SHA1_checksum,
    NULL
};

#ifdef DES3_OLD_ENCTYPE
struct _krb5_encryption_type _krb5_enctype_des3_cbc_md5 = {
    ETYPE_DES3_CBC_MD5,
    "des3-cbc-md5",
    8,
    8,
    8,
    &keytype_des3,
    &_krb5_checksum_rsa_md5,
    &_krb5_checksum_rsa_md5_des3,
    F_WARNING,
    _krb5_evp_encrypt,
    0,
    NULL
};
#endif

struct _krb5_encryption_type _krb5_enctype_des3_cbc_sha1 = {
    ETYPE_DES3_CBC_SHA1,
    "des3-cbc-sha1",
    8,
    8,
    8,
    &keytype_des3_derived,
    &_krb5_checksum_sha1,
    &_krb5_checksum_hmac_sha1_des3,
    F_DERIVED|F_WARNING,
    _krb5_evp_encrypt,
    16,
    DES3_prf
};

#ifdef DES3_OLD_ENCTYPE
struct _krb5_encryption_type _krb5_enctype_old_des3_cbc_sha1 = {
    ETYPE_OLD_DES3_CBC_SHA1,
    "old-des3-cbc-sha1",
    8,
    8,
    8,
    &keytype_des3,
    &_krb5_checksum_sha1,
    &_krb5_checksum_hmac_sha1_des3,
    F_WARNING,
    _krb5_evp_encrypt,
    0,
    NULL
};
#endif

struct _krb5_encryption_type _krb5_enctype_des3_cbc_none = {
    ETYPE_DES3_CBC_NONE,
    "des3-cbc-none",
    8,
    8,
    0,
    &keytype_des3_derived,
    &_krb5_checksum_none,
    NULL,
    F_PSEUDO|F_WARNING,
    _krb5_evp_encrypt,
    0,
    NULL
};

void
_krb5_DES3_random_to_key(krb5_context context,
			 krb5_keyblock *key,
			 const void *data,
			 size_t size)
{
    unsigned char *x;
    const uint8_t *q = data;
    uint8_t *k;
    int i, j;

    if (size < 21)
	abort();

    memset(key->keyvalue.data, 0, key->keyvalue.length);
    x = key->keyvalue.data;
    for (i = 0; i < 3; ++i) {
	unsigned char foo;
	for (j = 0; j < 7; ++j) {
	    unsigned char b = q[7 * i + j];

	    x[8 * i + j] = b;
	}
	foo = 0;
	for (j = 6; j >= 0; --j) {
	    foo |= q[7 * i + j] & 1;
	    foo <<= 1;
	}
	x[8 * i + 7] = foo;
    }
    k = key->keyvalue.data;
    for (i = 0; i < 3; i++) {
	CCDesSetOddParity(&k[i * 8], 8);
	if(CCDesIsWeakKey(&k[i * 8], 8))
	    _krb5_xor((void *)&k[i * 8],
		      (const unsigned char*)"\0\0\0\0\0\0\0\xf0");
    }
}

#endif /* HEIM_KRB5_DES3 */
