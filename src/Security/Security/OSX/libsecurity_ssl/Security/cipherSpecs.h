/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 3, 2022.
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
/*
 * cipherSpecs.h - SSLCipherSpec declarations
 */

#ifndef	_CIPHER_SPECS_H_
#define _CIPHER_SPECS_H_

#include <stdint.h>
#include "CipherSuite.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Cipher Algorithm */
typedef enum {
    SSL_CipherAlgorithmNull,
    SSL_CipherAlgorithmRC2_128,
    SSL_CipherAlgorithmRC4_128,
    SSL_CipherAlgorithmDES_CBC,
    SSL_CipherAlgorithm3DES_CBC,
    SSL_CipherAlgorithmAES_128_CBC,
    SSL_CipherAlgorithmAES_256_CBC,
    SSL_CipherAlgorithmAES_128_GCM,
    SSL_CipherAlgorithmAES_256_GCM,
} SSL_CipherAlgorithm;

/* The HMAC algorithms we support */
typedef enum {
    HA_Null = 0,		// i.e., uninitialized
    HA_SHA1,
    HA_MD5,
    HA_SHA256,
    HA_SHA384
} HMAC_Algs;

typedef enum
{   SSL_NULL_auth,
    SSL_RSA,
    SSL_RSA_EXPORT,
    SSL_DH_DSS,
    SSL_DH_DSS_EXPORT,
    SSL_DH_RSA,
    SSL_DH_RSA_EXPORT,
    SSL_DHE_DSS,
    SSL_DHE_DSS_EXPORT,
    SSL_DHE_RSA,
    SSL_DHE_RSA_EXPORT,
    SSL_DH_anon,
    SSL_DH_anon_EXPORT,
    SSL_Fortezza,

    /* ECDSA addenda, RFC 4492 */
    SSL_ECDH_ECDSA,
    SSL_ECDHE_ECDSA,
    SSL_ECDH_RSA,
    SSL_ECDHE_RSA,
    SSL_ECDH_anon,

    /* PSK, RFC 4279 */
    TLS_PSK,
    TLS_DHE_PSK,
    TLS_RSA_PSK,
    
} KeyExchangeMethod;


HMAC_Algs sslCipherSuiteGetMacAlgorithm(SSLCipherSuite cipherSuite);
SSL_CipherAlgorithm sslCipherSuiteGetSymmetricCipherAlgorithm(SSLCipherSuite cipherSuite);
KeyExchangeMethod sslCipherSuiteGetKeyExchangeMethod(SSLCipherSuite cipherSuite);

uint8_t sslCipherSuiteGetMacSize(SSLCipherSuite cipherSuite);
uint8_t sslCipherSuiteGetSymmetricCipherKeySize(SSLCipherSuite cipherSuite);
uint8_t sslCipherSuiteGetSymmetricCipherBlockIvSize(SSLCipherSuite cipherSuite);

/*
 * Determine if an SSLCipherSuite is SSLv2 only.
 */
#define CIPHER_SUITE_IS_SSLv2(suite)	((suite & 0xff00) == 0xff00)

#ifdef __cplusplus
}
#endif

#endif	/* _CIPHER_SPECS_H_ */
