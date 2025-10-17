/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 20, 2024.
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
#include "feeHash.h"
#include "ckMD5.h"
#include "falloc.h"
#include "platform.h"

/*
 * Private data for this object. A feeHash handle is cast to aa pointer
 * to one of these.
 */
typedef struct {
	MD5Context		context;
	int 			isDone;
	unsigned char 	digest[MD5_DIGEST_SIZE];
} hashInst;

/*
 * Alloc and init an empty hash object.
 */
feeHash feeHashAlloc(void)
{
	hashInst *hinst;

	hinst = (hashInst *) fmalloc(sizeof(hashInst));
	MD5Init(&hinst->context);
	hinst->isDone = 0;
	return hinst;
}

void feeHashReinit(feeHash hash)
{
	hashInst *hinst = (hashInst *) hash;

	MD5Init(&hinst->context);
	hinst->isDone = 0;
}

/*
 * Free a hash object.
 */
void feeHashFree(feeHash hash)
{
	hashInst *hinst = (hashInst *) hash;

	memset(hinst, 0, sizeof(hashInst));
	ffree(hinst);
}

/*
 * Add some data to the hash object.
 */
void feeHashAddData(feeHash hash,
	const unsigned char *data,
	unsigned dataLen)
{
	hashInst *hinst = (hashInst *) hash;

	if(hinst->isDone) {
		/*
		 * Log some kind of error here...
		 */
		return;
	}
	MD5Update(&hinst->context, data, dataLen);
}

/*
 * Obtain a pointer to completed message digest, and the length of the digest.
 */
unsigned char *feeHashDigest(feeHash hash)
{
	hashInst *hinst = (hashInst *) hash;

	if(!hinst->isDone) {
		MD5Final(&hinst->context, hinst->digest);
		hinst->isDone = 1;
	}
	return hinst->digest;
}

unsigned feeHashDigestLen(void)
{
	return MD5_DIGEST_SIZE;
}

