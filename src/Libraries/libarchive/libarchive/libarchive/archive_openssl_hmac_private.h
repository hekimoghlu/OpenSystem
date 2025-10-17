/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 2, 2024.
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
#ifndef ARCHIVE_OPENSSL_HMAC_PRIVATE_H_INCLUDED
#define ARCHIVE_OPENSSL_HMAC_PRIVATE_H_INCLUDED

#ifndef __LIBARCHIVE_BUILD
#error This header is only to be used internally to libarchive.
#endif

#include <openssl/hmac.h>
#include <openssl/opensslv.h>

#if OPENSSL_VERSION_NUMBER < 0x10100000L || \
	(defined(LIBRESSL_VERSION_NUMBER) && LIBRESSL_VERSION_NUMBER < 0x20700000L)
#include <stdlib.h> /* malloc, free */
#include <string.h> /* memset */
static inline HMAC_CTX *HMAC_CTX_new(void)
{
	HMAC_CTX *ctx = (HMAC_CTX *)calloc(1, sizeof(HMAC_CTX));
	return ctx;
}

static inline void HMAC_CTX_free(HMAC_CTX *ctx)
{
	HMAC_CTX_cleanup(ctx);
	memset(ctx, 0, sizeof(*ctx));
	free(ctx);
}
#endif

#endif
