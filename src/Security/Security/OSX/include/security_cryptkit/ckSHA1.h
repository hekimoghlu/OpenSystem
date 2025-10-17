/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 19, 2023.
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
#ifndef	_CK_SHA1_H_
#define _CK_SHA1_H_

#if	!defined(__MACH__)
#include <feeTypes.h>
#else
#include <security_cryptkit/feeTypes.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Opaque sha1 object handle.
 */
typedef void *sha1Obj;

/*
 * Alloc and init an empty sha1 object.
 */
sha1Obj sha1Alloc(void);

/*
 * reinitialize an sha1 object for reuse.
 */
void sha1Reinit(sha1Obj sha1);

/*
 * Free an sha1 object.
 */
void sha1Free(sha1Obj sha1);

/*
 * Add some data to the sha1 object.
 */
void sha1AddData(sha1Obj sha1,
	const unsigned char *data,
	unsigned dataLen);

/*
 * Obtain a pointer to completed message digest. This disables further calls
 * to sha1AddData(). This pointer is NOT malloc'd; the associated data
 * persists only as long as this object does.
 */
unsigned char *sha1Digest(sha1Obj sha1);

/*
 * Obtain the length of the message digest.
 */
unsigned sha1DigestLen(void);

#ifdef __cplusplus
}
#endif

#endif	/*_CK_SHA1_H_*/
