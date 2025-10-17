/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 3, 2022.
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
#include <libkern/crypto/crypto_internal.h>
#include <libkern/crypto/sha2.h>
#include <libkern/libkern.h>
#include <kern/debug.h>
#include <corecrypto/ccdigest.h>

#if defined(CRYPTO_SHA2)

void
SHA256_Init(SHA256_CTX *ctx)
{
	const struct ccdigest_info *di;
	di = g_crypto_funcs->ccsha256_di;

	/* Make sure the context size for the digest info fits in the one we have */
	if (ccdigest_di_size(di) > sizeof(SHA256_CTX)) {
		panic("%s: inconsistent size for SHA256 context", __FUNCTION__);
	}

	g_crypto_funcs->ccdigest_init_fn(di, ctx->ctx);
}

void
SHA256_Update(SHA256_CTX *ctx, const void *data, size_t len)
{
	const struct ccdigest_info *di;
	di = g_crypto_funcs->ccsha256_di;

	g_crypto_funcs->ccdigest_update_fn(di, ctx->ctx, len, data);
}

void
SHA256_Final(void *digest, SHA256_CTX *ctx)
{
	const struct ccdigest_info *di;
	di = g_crypto_funcs->ccsha256_di;

	ccdigest_final(di, ctx->ctx, digest);
}

void
SHA384_Init(SHA384_CTX *ctx)
{
	const struct ccdigest_info *di;
	di = g_crypto_funcs->ccsha384_di;

	/* Make sure the context size for the digest info fits in the one we have */
	if (ccdigest_di_size(di) > sizeof(SHA384_CTX)) {
		panic("%s: inconsistent size for SHA384 context", __FUNCTION__);
	}

	g_crypto_funcs->ccdigest_init_fn(di, ctx->ctx);
}

void
SHA384_Update(SHA384_CTX *ctx, const void *data, size_t len)
{
	const struct ccdigest_info *di;
	di = g_crypto_funcs->ccsha384_di;

	g_crypto_funcs->ccdigest_update_fn(di, ctx->ctx, len, data);
}


void
SHA384_Final(void *digest, SHA384_CTX *ctx)
{
	const struct ccdigest_info *di;
	di = g_crypto_funcs->ccsha384_di;

	ccdigest_final(di, ctx->ctx, digest);
}

void
SHA512_Init(SHA512_CTX *ctx)
{
	const struct ccdigest_info *di;
	di = g_crypto_funcs->ccsha512_di;

	/* Make sure the context size for the digest info fits in the one we have */
	if (ccdigest_di_size(di) > sizeof(SHA512_CTX)) {
		panic("%s: inconsistent size for SHA512 context", __FUNCTION__);
	}

	g_crypto_funcs->ccdigest_init_fn(di, ctx->ctx);
}

void
SHA512_Update(SHA512_CTX *ctx, const void *data, size_t len)
{
	const struct ccdigest_info *di;
	di = g_crypto_funcs->ccsha512_di;

	g_crypto_funcs->ccdigest_update_fn(di, ctx->ctx, len, data);
}

void
SHA512_Final(void *digest, SHA512_CTX *ctx)
{
	const struct ccdigest_info *di;
	di = g_crypto_funcs->ccsha512_di;

	ccdigest_final(di, ctx->ctx, digest);
}

#else

/* As these are part of the KPI, we need to stub them out for any kernel configuration that does not support SHA2. */

void UNSUPPORTED_API(SHA256_Init, SHA256_CTX *ctx);
void UNSUPPORTED_API(SHA384_Init, SHA384_CTX *ctx);
void UNSUPPORTED_API(SHA512_Init, SHA512_CTX *ctx);
void UNSUPPORTED_API(SHA256_Update, SHA256_CTX *ctx, const void *data, size_t len);
void UNSUPPORTED_API(SHA384_Update, SHA384_CTX *ctx, const void *data, size_t len);
void UNSUPPORTED_API(SHA512_Update, SHA512_CTX *ctx, const void *data, size_t len);
void UNSUPPORTED_API(SHA256_Final, void *digest, SHA256_CTX *ctx);
void UNSUPPORTED_API(SHA384_Final, void *digest, SHA384_CTX *ctx);
void UNSUPPORTED_API(SHA512_Final, void *digest, SHA512_CTX *ctx);

#endif
