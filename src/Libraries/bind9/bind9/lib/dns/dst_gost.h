/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 10, 2022.
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
#ifndef DST_GOST_H
#define DST_GOST_H 1

#include <isc/lang.h>
#include <isc/log.h>
#include <dst/result.h>

#define ISC_GOST_DIGESTLENGTH 32U

#ifdef HAVE_OPENSSL_GOST
#include <openssl/evp.h>

typedef struct {
	EVP_MD_CTX *ctx;
#if OPENSSL_VERSION_NUMBER < 0x10100000L
	EVP_MD_CTX _ctx;
#endif
} isc_gost_t;

#endif
#ifdef HAVE_PKCS11_GOST
#include <pk11/pk11.h>

typedef pk11_context_t isc_gost_t;
#endif

ISC_LANG_BEGINDECLS

#if defined(HAVE_OPENSSL_GOST) || defined(HAVE_PKCS11_GOST)

isc_result_t
isc_gost_init(isc_gost_t *ctx);

void
isc_gost_invalidate(isc_gost_t *ctx);

isc_result_t
isc_gost_update(isc_gost_t *ctx, const unsigned char *data, unsigned int len);

isc_result_t
isc_gost_final(isc_gost_t *ctx, unsigned char *digest);

ISC_LANG_ENDDECLS

#endif /* HAVE_OPENSSL_GOST || HAVE_PKCS11_GOST */

#endif /* DST_GOST_H */
