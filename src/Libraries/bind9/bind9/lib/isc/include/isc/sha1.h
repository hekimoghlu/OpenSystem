/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 29, 2025.
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
#ifndef ISC_SHA1_H
#define ISC_SHA1_H 1

/* $Id: sha1.h,v 1.19 2009/02/06 23:47:42 tbox Exp $ */

/*	$NetBSD: sha1.h,v 1.2 1998/05/29 22:55:44 thorpej Exp $	*/

/*! \file isc/sha1.h
 * \brief SHA-1 in C
 * \author By Steve Reid <steve@edmweb.com>
 * \note 100% Public Domain
 */

#include <isc/lang.h>
#include <isc/platform.h>
#include <isc/types.h>

#define ISC_SHA1_DIGESTLENGTH 20U
#define ISC_SHA1_BLOCK_LENGTH 64U

#ifdef ISC_PLATFORM_OPENSSLHASH
#include <openssl/opensslv.h>
#include <openssl/evp.h>

typedef struct {
	EVP_MD_CTX *ctx;
#if OPENSSL_VERSION_NUMBER < 0x10100000L
	EVP_MD_CTX _ctx;
#endif
} isc_sha1_t;

#elif PKCS11CRYPTO
#include <pk11/pk11.h>

typedef pk11_context_t isc_sha1_t;

#else

typedef struct {
	isc_uint32_t state[5];
	isc_uint32_t count[2];
	unsigned char buffer[ISC_SHA1_BLOCK_LENGTH];
} isc_sha1_t;
#endif

ISC_LANG_BEGINDECLS

void
isc_sha1_init(isc_sha1_t *ctx);

void
isc_sha1_invalidate(isc_sha1_t *ctx);

void
isc_sha1_update(isc_sha1_t *ctx, const unsigned char *data, unsigned int len);

void
isc_sha1_final(isc_sha1_t *ctx, unsigned char *digest);

ISC_LANG_ENDDECLS

#endif /* ISC_SHA1_H */
