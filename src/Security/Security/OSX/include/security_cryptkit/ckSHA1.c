/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 2, 2024.
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
#include "ckconfig.h"
#include "feeTypes.h"
#include "ckSHA1.h"

/*
 * For linking with AppleCSP: use libSystem SHA1 implementation.
 */
#include <CommonCrypto/CommonDigest.h>
#include "falloc.h"
#include "platform.h"

/*
 * Trivial wrapper for SHA_CTX; a sha1Obj is a pointer to this.
 */
typedef struct {
	CC_SHA1_CTX		ctx;
	unsigned char   digest[CC_SHA1_DIGEST_LENGTH];
} Sha1Obj;

sha1Obj sha1Alloc(void)
{
	void *rtn = fmalloc(sizeof(Sha1Obj));
	memset(rtn, 0, sizeof(Sha1Obj));
	CC_SHA1_Init(&(((Sha1Obj *)rtn)->ctx));
	return (sha1Obj)rtn;
}

void sha1Reinit(sha1Obj sha1)
{
	Sha1Obj *ctx = (Sha1Obj *)sha1;
	CC_SHA1_Init(&ctx->ctx);
}

void sha1Free(sha1Obj sha1)
{
	memset(sha1, 0, sizeof(Sha1Obj));
	ffree(sha1);
}

void sha1AddData(sha1Obj sha1,
	const unsigned char *data,
	unsigned dataLen)
{
	Sha1Obj *ctx = (Sha1Obj *)sha1;
	CC_SHA1_Update(&ctx->ctx, data, dataLen);
}

unsigned char *sha1Digest(sha1Obj sha1)
{
	Sha1Obj *ctx = (Sha1Obj *)sha1;
	CC_SHA1_Final(ctx->digest, &ctx->ctx);
	return ctx->digest;
}

unsigned sha1DigestLen(void)
{
	return CC_SHA1_DIGEST_LENGTH;
}

