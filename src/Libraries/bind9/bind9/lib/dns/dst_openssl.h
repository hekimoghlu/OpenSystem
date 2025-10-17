/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 22, 2024.
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
/* $Id: dst_openssl.h,v 1.11 2011/03/12 04:59:48 tbox Exp $ */

#ifndef DST_OPENSSL_H
#define DST_OPENSSL_H 1

#include <isc/lang.h>
#include <isc/log.h>
#include <isc/result.h>

#include <openssl/err.h>
#include <openssl/rand.h>
#include <openssl/evp.h>
#include <openssl/conf.h>
#include <openssl/crypto.h>
#include <openssl/bn.h>

#if !defined(OPENSSL_NO_ENGINE) && \
    ((defined(CRYPTO_LOCK_ENGINE) && \
      (OPENSSL_VERSION_NUMBER >= 0x0090707f)) || \
     (OPENSSL_VERSION_NUMBER >= 0x10100000L))
#define USE_ENGINE 1
#endif

#if OPENSSL_VERSION_NUMBER < 0x10100000L || defined(LIBRESSL_VERSION_NUMBER)
/*
 * These are new in OpenSSL 1.1.0.  BN_GENCB _cb needs to be declared in
 * the function like this before the BN_GENCB_new call:
 *
 * #if OPENSSL_VERSION_NUMBER < 0x10100000L
 *     	 _cb;
 * #endif
 */
#define BN_GENCB_free(x) (x = NULL);
#define BN_GENCB_new() (&_cb)
#define BN_GENCB_get_arg(x) ((x)->arg)
#endif

#if OPENSSL_VERSION_NUMBER >= 0x10100000L
/*
 * EVP_dss1() is a version of EVP_sha1() that was needed prior to
 * 1.1.0 because there was a link between digests and signing algorithms;
 * the link has been eliminated and EVP_sha1() can be used now instead.
 */
#define EVP_dss1 EVP_sha1
#endif

ISC_LANG_BEGINDECLS

isc_result_t
dst__openssl_toresult(isc_result_t fallback);

isc_result_t
dst__openssl_toresult2(const char *funcname, isc_result_t fallback);

isc_result_t
dst__openssl_toresult3(isc_logcategory_t *category,
		       const char *funcname, isc_result_t fallback);

#ifdef USE_ENGINE
ENGINE *
dst__openssl_getengine(const char *engine);
#else
#define dst__openssl_getengine(x) NULL
#endif

ISC_LANG_ENDDECLS

#endif /* DST_OPENSSL_H */
/*! \file */
