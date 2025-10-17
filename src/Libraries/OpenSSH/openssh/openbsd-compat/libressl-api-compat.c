/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 5, 2022.
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
#include "includes.h"

#ifdef WITH_OPENSSL

#include <sys/types.h>

#include <stdlib.h>
#include <string.h>

#include <openssl/evp.h>

#ifndef HAVE_EVP_CIPHER_CTX_GET_IV
int
EVP_CIPHER_CTX_get_iv(const EVP_CIPHER_CTX *ctx, unsigned char *iv, size_t len)
{
	if (ctx == NULL)
		return 0;
	if (EVP_CIPHER_CTX_iv_length(ctx) < 0)
		return 0;
	if (len != (size_t)EVP_CIPHER_CTX_iv_length(ctx))
		return 0;
	if (len > EVP_MAX_IV_LENGTH)
		return 0; /* sanity check; shouldn't happen */
	/*
	 * Skip the memcpy entirely when the requested IV length is zero,
	 * since the iv pointer may be NULL or invalid.
	 */
	if (len != 0) {
		if (iv == NULL)
			return 0;
# ifdef HAVE_EVP_CIPHER_CTX_IV
		memcpy(iv, EVP_CIPHER_CTX_iv(ctx), len);
# else
		memcpy(iv, ctx->iv, len);
# endif /* HAVE_EVP_CIPHER_CTX_IV */
	}
	return 1;
}
#endif /* HAVE_EVP_CIPHER_CTX_GET_IV */

#ifndef HAVE_EVP_CIPHER_CTX_SET_IV
int
EVP_CIPHER_CTX_set_iv(EVP_CIPHER_CTX *ctx, const unsigned char *iv, size_t len)
{
	if (ctx == NULL)
		return 0;
	if (EVP_CIPHER_CTX_iv_length(ctx) < 0)
		return 0;
	if (len != (size_t)EVP_CIPHER_CTX_iv_length(ctx))
		return 0;
	if (len > EVP_MAX_IV_LENGTH)
		return 0; /* sanity check; shouldn't happen */
	/*
	 * Skip the memcpy entirely when the requested IV length is zero,
	 * since the iv pointer may be NULL or invalid.
	 */
	if (len != 0) {
		if (iv == NULL)
			return 0;
# ifdef HAVE_EVP_CIPHER_CTX_IV_NOCONST
		memcpy(EVP_CIPHER_CTX_iv_noconst(ctx), iv, len);
# else
		memcpy(ctx->iv, iv, len);
# endif /* HAVE_EVP_CIPHER_CTX_IV_NOCONST */
	}
	return 1;
}
#endif /* HAVE_EVP_CIPHER_CTX_SET_IV */

#endif /* WITH_OPENSSL */
