/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 4, 2024.
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

#ifdef KRB5
#include <krb5-types.h>
#endif

#ifdef HAVE_COMMONCRYPTO_COMMONKEYDERIVATION_H
#include <CommonCrypto/CommonKeyDerivation.h>
#endif

#include <stdio.h>
#include <stdlib.h>

#include <evp.h>
#include <hmac.h>

#include <roken.h>

/**
 * As descriped in PKCS5, convert a password, salt, and iteration counter into a crypto key.
 *
 * @param password Password.
 * @param password_len Length of password.
 * @param salt Salt
 * @param salt_len Length of salt.
 * @param iter iteration counter.
 * @param keylen the output key length.
 * @param key the output key.
 *
 * @return 1 on success, non 1 on failure.
 *
 * @ingroup hcrypto_misc
 */

int
PKCS5_PBKDF2_HMAC_SHA1(const void * password, size_t password_len,
		       const void * salt, size_t salt_len,
		       unsigned long iter,
		       size_t keylen, void *key)
{
#ifdef HAVE_COMMONCRYPTO_COMMONKEYDERIVATION_H
    if (CCKeyDerivationPBKDF(kCCPBKDF2,
			     password ? password : "", password_len,
			     salt, salt_len,
			     kCCPRFHmacAlgSHA1, iter,
			     key, keylen) != 0)
	return 0;
    return 1;
#else
    size_t datalen, leftofkey, checksumsize;
    char *data, *tmpcksum;
    uint32_t keypart;
    const EVP_MD *md;
    unsigned long i;
    int j;
    char *p;
    unsigned int hmacsize;

    md = EVP_sha1();
    checksumsize = EVP_MD_size(md);
    datalen = salt_len + 4;

    tmpcksum = malloc(checksumsize + datalen);
    if (tmpcksum == NULL)
	return 0;

    data = &tmpcksum[checksumsize];

    memcpy(data, salt, salt_len);

    keypart = 1;
    leftofkey = keylen;
    p = key;

    while (leftofkey) {
	int len;

	if (leftofkey > checksumsize)
	    len = checksumsize;
	else
	    len = leftofkey;

	data[datalen - 4] = (keypart >> 24) & 0xff;
	data[datalen - 3] = (keypart >> 16) & 0xff;
	data[datalen - 2] = (keypart >> 8)  & 0xff;
	data[datalen - 1] = (keypart)       & 0xff;

	HMAC(md, password, password_len, data, datalen,
	     tmpcksum, &hmacsize);

	memcpy(p, tmpcksum, len);
	for (i = 1; i < iter; i++) {
	    HMAC(md, password, password_len, tmpcksum, checksumsize,
		 tmpcksum, &hmacsize);

	    for (j = 0; j < len; j++)
		p[j] ^= tmpcksum[j];
	}

	p += len;
	leftofkey -= len;
	keypart++;
    }

    free(tmpcksum);

    return 1;
#endif
}
