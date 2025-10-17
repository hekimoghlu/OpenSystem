/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 22, 2023.
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
 * sslCrypto.h - interface between SSL and crypto libraries
 */

#ifndef	_SSL_CRYPTO_H_
#define _SSL_CRYPTO_H_	1

#include "ssl.h"
#include "sslContext.h"
#include <Security/SecKeyPriv.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifndef		NDEBUG
extern void stPrintCdsaError(const char *op, OSStatus crtn);
#else
#define stPrintCdsaError(o, cr)
#endif

/*
 * Create a new SecTrust object and return it.
 */
OSStatus
sslCreateSecTrust(
	SSLContext				*ctx,
    SecTrustRef             *trust); 	/* RETURNED */

OSStatus sslVerifySelectedCipher(
	SSLContext 		*ctx);

/*
 * Set the pubkey after receiving the certificate
 */
int tls_set_peer_pubkey(SSLContext *ctx);

/*
 * Verify the peer cert chain (after receiving the server hello or client cert)
 */
int tls_verify_peer_cert(SSLContext *ctx);


#
#ifdef __cplusplus
}
#endif


#endif	/* _SSL_CRYPTO_H_ */
