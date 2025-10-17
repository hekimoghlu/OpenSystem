/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 9, 2024.
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
 * sslKeychain.h - Apple Keychain routines
 */

#ifndef	_SSL_KEYCHAIN_H_
#define _SSL_KEYCHAIN_H_


#include "sslContext.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Free the tls_private_key_t struct and associated SecKeyRef context that were created by parseIncomingCerts */
void sslFreePrivKey(tls_private_key_t *sslPrivKey);

/* Create a tls_private_key_t struct and SSLCertificate list, from a CFArray */
OSStatus
parseIncomingCerts(
	SSLContext			*ctx,
	CFArrayRef			certs,
	SSLCertificate      **destCertChain,/* &ctx->{localCertChain,encryptCertChain} */
    tls_private_key_t   *privKeyRef);	/* &ctx->signingPrivKeyRef, etc. */

#ifdef __cplusplus
}
#endif

#endif	/* _SSL_KEYCHAIN_H_ */
