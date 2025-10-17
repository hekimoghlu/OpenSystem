/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 1, 2023.
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
#include <config.h>

#include <stdio.h>
#include <stdlib.h>
#include <krb5-types.h>
#include <assert.h>

#include <rsa.h>

#ifdef HEIM_HC_SF

#include <Security/Security.h> 
#include <Security/SecRSAKey.h> 

#include <rfc2459_asn1.h>

#include "common.h"

/*
 *
 */

static SecKeyRef
CreateKeyFromRSA(RSA *rsa, int use_public)
{
    SecKeyRef (*CreateMethod)(CFAllocatorRef, const uint8_t *, CFIndex, SecKeyEncoding) = NULL;
    size_t size = 0, keylength;
    void *keydata;
    int ret;
	
    if (use_public) {
	RSAPublicKey k;

	CreateMethod = SecKeyCreateRSAPublicKey;

	memset(&k, 0, sizeof(k));

	ret = _hc_BN_to_integer(rsa->n, &k.modulus);
	if (ret == 0)
	    ret = _hc_BN_to_integer(rsa->e, &k.publicExponent);
	if (ret) {
	    free_RSAPublicKey(&k);
	    return NULL;
	}

	ASN1_MALLOC_ENCODE(RSAPublicKey, keydata, keylength, &k, &size, ret);
	free_RSAPublicKey(&k);
	if (ret)
	    return NULL;
	if (size != keylength)
	    abort();

    } else {
	RSAPrivateKey k;

	CreateMethod = SecKeyCreateRSAPrivateKey;

	memset(&k, 0, sizeof(k));

	k.version = 1;
	ret = _hc_BN_to_integer(rsa->n, &k.modulus);
	if (ret == 0)
	    ret = _hc_BN_to_integer(rsa->e, &k.publicExponent);
	if (ret == 0)
	    ret = _hc_BN_to_integer(rsa->d, &k.privateExponent);
	if (ret == 0)
	    ret = _hc_BN_to_integer(rsa->p, &k.prime1);
	if (ret == 0)
	    ret = _hc_BN_to_integer(rsa->q, &k.prime2);
	if (ret == 0)
	    ret = _hc_BN_to_integer(rsa->dmp1, &k.exponent1);
	if (ret == 0)
	    ret = _hc_BN_to_integer(rsa->dmq1, &k.exponent2);
	if (ret == 0)
	    ret = _hc_BN_to_integer(rsa->iqmp, &k.coefficient);
	if (ret) {
	    free_RSAPrivateKey(&k);
	    return NULL;
	}

	ASN1_MALLOC_ENCODE(RSAPrivateKey, keydata, keylength, &k, &size, ret);
	free_RSAPrivateKey(&k);
	if (ret)
	    return NULL;
	if (size != keylength)
	    abort();
    }

    SecKeyRef key = CreateMethod(NULL, keydata, keylength, kSecKeyEncodingPkcs1);
    free(keydata);
    return key;
}

/*
 *
 */

static int
sf_rsa_public_encrypt(int flen, const unsigned char* from,
		      unsigned char* to, RSA* rsa, int padding)
{
    SecKeyRef key = CreateKeyFromRSA(rsa, 1);
    OSStatus status;
    size_t tlen = RSA_size(rsa);

    if (key == NULL)
	return -1;

    if (padding != RSA_PKCS1_PADDING)
	return -1;

    status = SecKeyEncrypt(key, kSecPaddingPKCS1, from, flen, to, &tlen);
    CFRelease(key);
    if (status)
	return -1;
    if (tlen > (size_t)RSA_size(rsa))
	abort();
    return tlen;
}

static int
sf_rsa_public_decrypt(int flen, const unsigned char* from,
		      unsigned char* to, RSA* rsa, int padding)
{
    SecKeyRef key = CreateKeyFromRSA(rsa, 1);
    OSStatus status;
    size_t tlen = RSA_size(rsa);

    if (key == NULL)
	return -1;

    if (padding != RSA_PKCS1_PADDING)
	return -1;

    /* SecKeyDecrypt gets decrytion wrong for public keys in the PKCS1 case (14322412), lets do PKCS1 (un)padding ourself */
    status = SecKeyDecrypt(key, kSecPaddingNone, from, flen, to, &tlen);
    CFRelease(key);
    if (status)
	return -1;
    if (tlen > (size_t)RSA_size(rsa))
	abort();

    unsigned char *p = to;

    if (tlen < 1)
	return -1;
    if (*p != 1)
	return -1;
    tlen--; p++;
    while (tlen && *p == 0xff) {
	tlen--; p++;
    }
    if (tlen == 0 || *p != 0)
	return -1;
    tlen--; p++;

    memmove(to, p, tlen);

    return tlen;
}

static int
sf_rsa_private_encrypt(int flen, const unsigned char* from,
		       unsigned char* to, RSA* rsa, int padding)
{
    SecKeyRef key = CreateKeyFromRSA(rsa, 0);
    OSStatus status;
    size_t tlen = RSA_size(rsa);

    if (key == NULL)
	return -1;

    if (padding != RSA_PKCS1_PADDING)
	return -1;

    status = SecKeyEncrypt(key, kSecPaddingPKCS1, from, flen, to, &tlen);
    CFRelease(key);
    if (status)
	return -1;
    if (tlen > (size_t)RSA_size(rsa))
	abort();
    return tlen;
}

static int
sf_rsa_private_decrypt(int flen, const unsigned char* from,
		       unsigned char* to, RSA* rsa, int padding)
{
    SecKeyRef key = CreateKeyFromRSA(rsa, 0);
    OSStatus status;
    size_t tlen = RSA_size(rsa);

    if (key == NULL)
	return -1;

    if (padding != RSA_PKCS1_PADDING)
	return -1;

    /* SecKeyDecrypt gets kSecPaddingPKCS1 wrong (14322412), lets inline pkcs1 padding here ourself */

    status = SecKeyDecrypt(key, kSecPaddingNone, from, flen, to, &tlen);
    CFRelease(key);
    if (status)
	return -1;
    if (tlen > (size_t)RSA_size(rsa))
	abort();

    unsigned char *p = to;

    if (tlen < 1) return -1;

    if (*p != 1)
	return -1;
    tlen--; p++;
    while (tlen && *p == 0xff) {
	tlen--; p++;
    }
    if (tlen == 0 || *p != 0)
	return -1;
    tlen--; p++;

    memmove(to, p, tlen);

    return tlen;
}


static int
sf_rsa_generate_key(RSA *rsa, int bits, BIGNUM *e, BN_GENCB *cb)
{
    return -1;
}

static int
sf_rsa_init(RSA *rsa)
{
    return 1;
}

static int
sf_rsa_finish(RSA *rsa)
{
    return 1;
}

const RSA_METHOD _hc_rsa_sf_method = {
    "hcrypto sf RSA",
    sf_rsa_public_encrypt,
    sf_rsa_public_decrypt,
    sf_rsa_private_encrypt,
    sf_rsa_private_decrypt,
    NULL,
    NULL,
    sf_rsa_init,
    sf_rsa_finish,
    0,
    NULL,
    NULL,
    NULL,
    sf_rsa_generate_key
};
#endif

const RSA_METHOD *
RSA_sf_method(void)
{
#ifdef HEIM_HC_SF
    return &_hc_rsa_sf_method;
#else
    return NULL;
#endif
}
