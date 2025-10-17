/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 2, 2022.
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
 * sslBuildFlags.h - Common build flags
 */

#ifndef	_SSL_BUILD_FLAGS_H_
#define _SSL_BUILD_FLAGS_H_				1

#if defined(__cplusplus)
extern "C" {
#endif

/*
 * Implementation-specific functionality.
 */
#if 1
#undef USE_CDSA_CRYPTO				/* use corecrypto, instead of CDSA */
#undef USE_SSLCERTIFICATE			/* use CF-based certs, not structs */
#endif

/*
 * Work around the Netscape Server Key Exchange bug. When this is
 * true, only do server key exchange if both of the following are
 * true:
 *
 *   -- an export-grade ciphersuite has been negotiated, and
 *   -- an encryptPrivKey is present in the context
 */
#define SSL_SERVER_KEYEXCH_HACK		0

/*
 * RSA functions which use a public key to do encryption force
 * the proper usage bit because the CL always gives us
 * a pub key (from a cert) with only the verify bit set.
 * This needs a mod to the CL to do the right thing, and that
 * might not be enough - what if server certs don't have the
 * appropriate usage bits?
 */
#define RSA_PUB_KEY_USAGE_HACK		1

/* debugging flags */
#ifdef	NDEBUG
#define SSL_DEBUG					0
#define ERROR_LOG_ENABLE			0
#else
#define SSL_DEBUG					1
#define ERROR_LOG_ENABLE			1
#endif	/* NDEBUG */

/*
 * Server-side PAC-based EAP support currently enabled only for debug builds.
 */
#ifdef	NDEBUG
#define SSL_PAC_SERVER_ENABLE		0
#else
#define SSL_PAC_SERVER_ENABLE		1
#endif

#define ENABLE_SSLV2                0

/* Experimental */
#define ENABLE_DTLS                 1

#define ENABLE_3DES                 1		/* normally enabled */
#define ENABLE_DES                  0		/* normally disabled */
#define ENABLE_RC2                  0		/* normally disabled */
#define ENABLE_AES                  1		/* normally enabled, our first preference */    
#define ENABLE_AES256               1		/* normally enabled */
#define ENABLE_ECDHE                1
#define ENABLE_ECDHE_RSA            1
#define ENABLE_ECDH                 1
#define ENABLE_ECDH_RSA             1

#if defined(__cplusplus)
}
#endif

#endif	/* _SSL_BUILD_FLAGS_H_ */
