/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 15, 2024.
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
#include <libkern/crypto/md5.h>
#include <kern/debug.h>
#include <corecrypto/ccdigest.h>

static uint64_t
getCount(MD5_CTX *ctx)
{
	return (((uint64_t)ctx->count[0]) << 32) | (ctx->count[1]);
}

static void
setCount(MD5_CTX *ctx, uint64_t count)
{
	ctx->count[0] = (uint32_t)(count >> 32);
	ctx->count[1] = (uint32_t)count;
}

/* Copy a ccdigest ctx into a legacy MD5 context */
static void
DiToMD5(const struct ccdigest_info *di, struct ccdigest_ctx *di_ctx, MD5_CTX *md5_ctx)
{
	setCount(md5_ctx, ccdigest_nbits(di, di_ctx) / 8 + ccdigest_num(di, di_ctx));
	memcpy(md5_ctx->buffer, ccdigest_data(di, di_ctx), di->block_size);
	memcpy(md5_ctx->state, ccdigest_state_ccn(di, di_ctx), di->state_size);
}

/* Copy a legacy MD5 context into a ccdigest ctx  */
static void
MD5ToDi(const struct ccdigest_info *di, MD5_CTX *md5_ctx, struct ccdigest_ctx *di_ctx)
{
	uint64_t count = getCount(md5_ctx);

	ccdigest_num(di, di_ctx) = (unsigned)(count % di->block_size);
	ccdigest_nbits(di, di_ctx) = (count - ccdigest_num(di, di_ctx)) * 8;
	memcpy(ccdigest_data(di, di_ctx), md5_ctx->buffer, di->block_size);
	memcpy(ccdigest_state_ccn(di, di_ctx), md5_ctx->state, di->state_size);
}

void
MD5Init(MD5_CTX *ctx)
{
	const struct ccdigest_info *di = g_crypto_funcs->ccmd5_di;
	ccdigest_di_decl(di, di_ctx);

	g_crypto_funcs->ccdigest_init_fn(di, di_ctx);

	DiToMD5(di, di_ctx, ctx);
}

void
MD5Update(MD5_CTX *ctx, const void *data, unsigned int len)
{
	const struct ccdigest_info *di = g_crypto_funcs->ccmd5_di;
	ccdigest_di_decl(di, di_ctx);

	MD5ToDi(di, ctx, di_ctx);
	g_crypto_funcs->ccdigest_update_fn(di, di_ctx, len, data);
	DiToMD5(di, di_ctx, ctx);
}

void
MD5Final(unsigned char digest[MD5_DIGEST_LENGTH], MD5_CTX *ctx)
{
	const struct ccdigest_info *di = g_crypto_funcs->ccmd5_di;
	ccdigest_di_decl(di, di_ctx);

	MD5ToDi(di, ctx, di_ctx);
	ccdigest_final(di, di_ctx, digest);
}
