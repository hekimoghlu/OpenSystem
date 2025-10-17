/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 29, 2021.
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
#ifndef	_CK_FALLOC_H_
#define _CK_FALLOC_H_

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Clients can *optionally* register external memory alloc/free functions here.
 */
typedef void *(mallocExternFcn)(unsigned size);
typedef void (freeExternFcn)(void *data);
typedef void *(reallocExternFcn)(void *oldData, unsigned newSize);
void fallocRegister(mallocExternFcn *mallocExtern,
	freeExternFcn *freeExtern,
	reallocExternFcn *reallocExtern);
	
	
void *fmalloc(unsigned size);		/* general malloc */
void *fmallocWithData(const void *origData,
	unsigned origDataLen);		/* malloc, copy existing data */
void ffree(void *data);			/* general free */
void *frealloc(void *oldPtr, unsigned newSize);

#ifdef __cplusplus
}
#endif

#endif	/*_CK_FALLOC_H_*/
