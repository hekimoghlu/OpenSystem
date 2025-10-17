/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 27, 2021.
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
#ifndef	_CK_FEEHASH_H_
#define _CK_FEEHASH_H_

#if	!defined(__MACH__)
#include <ckconfig.h>
#include <feeTypes.h>
#else
#include <security_cryptkit/ckconfig.h>
#include <security_cryptkit/feeTypes.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Opaque hash object handle.
 */
typedef void *feeHash;

/*
 * Alloc and init an empty hash object.
 */
feeHash feeHashAlloc(void);

/*
 * reinitialize a hash object for reuse.
 */
void feeHashReinit(feeHash hash);

/*
 * Free a hash object.
 */
void feeHashFree(feeHash hash);

/*
 * Add some data to the hash object.
 */
void feeHashAddData(feeHash hash,
	const unsigned char *data,
	unsigned dataLen);

/*
 * Obtain a pointer to completed message digest. This disables further calls
 * to feeHashAddData(). This pointer is NOT malloc'd; the associated data
 * persists only as long as this object does.
 */
unsigned char *feeHashDigest(feeHash hash);

/*
 * Obtain the length of the message digest.
 */
unsigned feeHashDigestLen(void);

#ifdef __cplusplus
}
#endif

#endif	/*_CK_FEEHASH_H_*/
