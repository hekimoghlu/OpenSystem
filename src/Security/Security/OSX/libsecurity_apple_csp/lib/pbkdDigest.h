/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 25, 2025.
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
 * pbkdDigest.h - SHA1/MD5 digest object for HMAC and PBE routines
 */
 
#ifndef	_PBKD_DIGEST_H_
#define _PBKD_DIGEST_H_

#include <Security/cssmtype.h>
#include <CommonCrypto/CommonDigest.h>

#ifdef __cplusplus
extern "C" {
#endif

#define kSHA1DigestSize  		CC_SHA1_DIGEST_LENGTH
#define kSHA1BlockSize  		CC_SHA1_BLOCK_BYTES

#define kMD5DigestSize  		CC_MD5_DIGEST_LENGTH
#define kMD5BlockSize  			CC_MD5_BLOCK_BYTES

#define kMD2DigestSize  		CC_MD2_DIGEST_LENGTH
#define kMD2BlockSize  			CC_MD2_BLOCK_BYTES

#define kMaxDigestSize			kSHA1DigestSize

typedef int (*DigestInitFcn)(void *ctx);
typedef int (*DigestUpdateFcn)(void *ctx, const void *data, unsigned long len);
typedef int (*DigestFinalFcn)(void *md, void *c);

/* callouts to eay/libmd implementations */
typedef struct {
	DigestInitFcn		init;
	DigestUpdateFcn		update;
	DigestFinalFcn		final;
} DigestOps;

typedef	struct {
	union {
		CC_SHA1_CTX 	sha1Context;
		CC_MD5_CTX		md5Context;
		CC_MD2_CTX		md2Context;
	} dig;
	DigestOps 		*ops;
	CSSM_ALGORITHMS hashAlg;
} DigestCtx;

/* Ops on a DigestCtx - all return zero on error, like the underlying digests do */
int DigestCtxInit(
	DigestCtx		*ctx,
	CSSM_ALGORITHMS hashAlg);
void DigestCtxFree(
	DigestCtx 	*ctx);
int DigestCtxUpdate(
	DigestCtx 	*ctx,
	const void 	*textPtr,
	uint32 textLen);
int DigestCtxFinal(
	DigestCtx 	*ctx,
	void 		*digest);

#ifdef __cplusplus
}
#endif

#endif	/* _PBKD_DIGEST_H_ */

