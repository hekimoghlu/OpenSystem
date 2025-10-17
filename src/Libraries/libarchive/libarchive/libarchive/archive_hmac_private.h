/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 1, 2025.
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
#ifndef ARCHIVE_HMAC_PRIVATE_H_INCLUDED
#define ARCHIVE_HMAC_PRIVATE_H_INCLUDED

#ifndef __LIBARCHIVE_BUILD
#error This header is only to be used internally to libarchive.
#endif
/*
 * On systems that do not support any recognized crypto libraries,
 * the archive_hmac.c file is expected to define no usable symbols.
 *
 * But some compilers and linkers choke on empty object files, so
 * define a public symbol that will always exist.  This could
 * be removed someday if this file gains another always-present
 * symbol definition.
 */
int __libarchive_hmac_build_hack(void);

#ifdef __APPLE__
# include <AvailabilityMacros.h>
# if MAC_OS_X_VERSION_MAX_ALLOWED >= 1060
#  define ARCHIVE_HMAC_USE_Apple_CommonCrypto
# endif
#endif

#ifdef ARCHIVE_HMAC_USE_Apple_CommonCrypto
#include <CommonCrypto/CommonHMAC.h>

typedef	CCHmacContext archive_hmac_sha1_ctx;

#elif defined(_WIN32) && !defined(__CYGWIN__) && defined(HAVE_BCRYPT_H)
#include <bcrypt.h>

typedef struct {
	BCRYPT_ALG_HANDLE	hAlg;
	BCRYPT_HASH_HANDLE	hHash;
	DWORD				hash_len;
	PBYTE				hash;

} archive_hmac_sha1_ctx;

#elif defined(HAVE_LIBMBEDCRYPTO) && defined(HAVE_MBEDTLS_MD_H)
#include <mbedtls/md.h>

typedef mbedtls_md_context_t archive_hmac_sha1_ctx;

#elif defined(HAVE_LIBNETTLE) && defined(HAVE_NETTLE_HMAC_H)
#include <nettle/hmac.h>

typedef	struct hmac_sha1_ctx archive_hmac_sha1_ctx;

#elif defined(HAVE_LIBCRYPTO)
#include <openssl/opensslv.h>
#include <openssl/hmac.h>
#if OPENSSL_VERSION_NUMBER >= 0x30000000L
#include <openssl/params.h>

typedef EVP_MAC_CTX *archive_hmac_sha1_ctx;

#else
#include "archive_openssl_hmac_private.h"

typedef	HMAC_CTX* archive_hmac_sha1_ctx;
#endif

#else

typedef int archive_hmac_sha1_ctx;

#endif


/* HMAC */
#define archive_hmac_sha1_init(ctx, key, key_len)\
	__archive_hmac.__hmac_sha1_init(ctx, key, key_len)
#define archive_hmac_sha1_update(ctx, data, data_len)\
	__archive_hmac.__hmac_sha1_update(ctx, data, data_len)
#define archive_hmac_sha1_final(ctx, out, out_len)\
  	__archive_hmac.__hmac_sha1_final(ctx, out, out_len)
#define archive_hmac_sha1_cleanup(ctx)\
	__archive_hmac.__hmac_sha1_cleanup(ctx)


struct archive_hmac {
	/* HMAC */
	int (*__hmac_sha1_init)(archive_hmac_sha1_ctx *, const uint8_t *,
		size_t);
	void (*__hmac_sha1_update)(archive_hmac_sha1_ctx *, const uint8_t *,
		size_t);
	void (*__hmac_sha1_final)(archive_hmac_sha1_ctx *, uint8_t *, size_t *);
	void (*__hmac_sha1_cleanup)(archive_hmac_sha1_ctx *);
};

extern const struct archive_hmac __archive_hmac;
#endif /* ARCHIVE_HMAC_PRIVATE_H_INCLUDED */
