/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 24, 2025.
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
#ifndef MD4_H
#define MD4_H

#define	MD4_DIGEST_LENGTH		16

#ifdef __APPLE__
#include <sys/param.h>

/*
 * Shim the MD4 implementation out to libmd, which in turn is largely just
 * a CommonDigest shim.  The intermediate hop is just some macro definitions
 * that are guaranteed to match what openrsync / openbsd expect and we won't
 * actually need to link against libmd, so the indirection costs us  nothing.
 */
#include <md4.h>
#include <limits.h>

#define MD4_Init(ctx)			MD4Init(ctx)

/*
 * Shim MD4_Update out manually, because we have a slight size mismatch.
 * CommonCrypto wants a CC_LONG, which is actually a uint32_t on all platforms,
 * but we use size_t everywhere.
 */
static inline void
MD4_Update(MD4_CTX *ctx, const void *data, size_t len)
{
	size_t resid;

	while (len != 0) {
		resid = MIN(len, UINT_MAX);

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
		MD4Update(ctx, data, resid);
#pragma clang diagnostic pop
		len -= resid;
		data += resid;
	}
}

#define	MD4_Final(digest, ctx)		MD4Final(digest, ctx)
#else
/* Any 32-bit or wider unsigned integer data type will do */
typedef unsigned int MD4_u32plus;

typedef struct {
	MD4_u32plus lo, hi;
	MD4_u32plus a, b, c, d;
	unsigned char buffer[64];
	MD4_u32plus block[16];
} MD4_CTX;

extern void MD4_Init(MD4_CTX *ctx);
extern void MD4_Update(MD4_CTX *ctx, const void *data, unsigned long size);
extern void MD4_Final(unsigned char *result, MD4_CTX *ctx);
#endif /* __APPLE__ */

#endif
