/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 27, 2023.
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
/* $Id: hmacmd5.h,v 1.14 2009/02/06 23:47:42 tbox Exp $ */

/*! \file isc/hmacmd5.h
 * \brief This is the header file for the HMAC-MD5 keyed hash algorithm
 * described in RFC2104.
 */

#ifndef ISC_HMACMD5_H
#define ISC_HMACMD5_H 1

#include <pk11/site.h>

#ifndef PK11_MD5_DISABLE

#include <isc/lang.h>
#include <isc/md5.h>
#include <isc/platform.h>
#include <isc/types.h>

#define ISC_HMACMD5_KEYLENGTH 64

#ifdef ISC_PLATFORM_OPENSSLHASH
#include <openssl/opensslv.h>
#include <openssl/hmac.h>

typedef struct {
	HMAC_CTX *ctx;
#if OPENSSL_VERSION_NUMBER < 0x10100000L
	HMAC_CTX _ctx;
#endif
} isc_hmacmd5_t;

#elif PKCS11CRYPTO
#include <pk11/pk11.h>

typedef pk11_context_t isc_hmacmd5_t;

#else

typedef struct {
	isc_md5_t md5ctx;
	unsigned char key[ISC_HMACMD5_KEYLENGTH];
} isc_hmacmd5_t;
#endif

ISC_LANG_BEGINDECLS

void
isc_hmacmd5_init(isc_hmacmd5_t *ctx, const unsigned char *key,
		 unsigned int len);

void
isc_hmacmd5_invalidate(isc_hmacmd5_t *ctx);

void
isc_hmacmd5_update(isc_hmacmd5_t *ctx, const unsigned char *buf,
		   unsigned int len);

void
isc_hmacmd5_sign(isc_hmacmd5_t *ctx, unsigned char *digest);

isc_boolean_t
isc_hmacmd5_verify(isc_hmacmd5_t *ctx, unsigned char *digest);

isc_boolean_t
isc_hmacmd5_verify2(isc_hmacmd5_t *ctx, unsigned char *digest, size_t len);

ISC_LANG_ENDDECLS

#endif /* !PK11_MD5_DISABLE */

#endif /* ISC_HMACMD5_H */
