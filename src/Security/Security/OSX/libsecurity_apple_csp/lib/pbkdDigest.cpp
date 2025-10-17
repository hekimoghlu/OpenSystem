/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 21, 2023.
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
 * pbkdDigest.cpp - SHA1/MD5 digest object for HMAC and PBE routines
 */

#include "pbkdDigest.h"
#include <Security/cssmerr.h>
#include <string.h>

/* the casts are necessary to cover the polymorphous context types */
DigestOps Md2Ops = { 
	(DigestInitFcn)CC_MD2_Init, 
	(DigestUpdateFcn)CC_MD2_Update, 
	(DigestFinalFcn)CC_MD2_Final
};
DigestOps Md5Ops = { 
	(DigestInitFcn)CC_MD5_Init, 
	(DigestUpdateFcn)CC_MD5_Update, 
	(DigestFinalFcn)CC_MD5_Final
};
DigestOps Sha1Ops = { 
	(DigestInitFcn)CC_SHA1_Init, 
	(DigestUpdateFcn)CC_SHA1_Update, 
	(DigestFinalFcn)CC_SHA1_Final
};

/* Ops on a DigestCtx - all return zero on error, like the underlying digests do */
int DigestCtxInit(
	DigestCtx 	*ctx,
	CSSM_ALGORITHMS hashAlg)
{
	switch(hashAlg) {
		case CSSM_ALGID_SHA1:
			ctx->ops = &Sha1Ops;
			break;
		case CSSM_ALGID_MD5:
			ctx->ops = &Md5Ops;
			break;
		case CSSM_ALGID_MD2:
			ctx->ops = &Md2Ops;
			break;
		default:
			return 0;
	}
	ctx->hashAlg = hashAlg;
	return ctx->ops->init(&ctx->dig);
}

void DigestCtxFree(
	DigestCtx 	*ctx)
{
	memset(ctx, 0, sizeof(DigestCtx));
}

int DigestCtxUpdate(
	DigestCtx 	*ctx,
	const void *textPtr,
	uint32 textLen)
{
	return ctx->ops->update(&ctx->dig, textPtr, textLen);
}

int DigestCtxFinal(
	DigestCtx 	*ctx,
	void 		*digest)
{
	return ctx->ops->final(digest, &ctx->dig);
}
