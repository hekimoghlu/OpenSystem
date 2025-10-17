/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 28, 2025.
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
#include "platform.h"
#include "falloc.h"
#include <stdlib.h>

/* watchpoint emulator */
#define FALLOC_WATCH	0
#if		FALLOC_WATCH
#include <stdio.h>
/* set these with debugger */
void *mallocWatchAddrs;
void *freeWatchAddrs;
#endif

/* if NULL, use our own */
static mallocExternFcn *mallocExt = NULL;
static freeExternFcn *freeExt = NULL;
static reallocExternFcn *reallocExt = NULL;

void fallocRegister(mallocExternFcn *mallocExtern,
	freeExternFcn *freeExtern,
	reallocExternFcn *reallocExtern)
{
	mallocExt = mallocExtern;
	freeExt = freeExtern;
	reallocExt = reallocExtern;
}

/*
 * All this can be optimized and tailored to specific platforms, of course...
 */

void *fmalloc(unsigned size)
{
	void *rtn;
	if(mallocExt != NULL) {
		rtn = (mallocExt)(size);
	}
	else {
		rtn = malloc(size);
	}
	#if		FALLOC_WATCH
	if(rtn == mallocWatchAddrs) {
		printf("====fmalloc watchpoint (0x%x) hit\n",
			(unsigned)mallocWatchAddrs);
	}
	#endif
	return rtn;
}

void *fmallocWithData(const void *origData,
	unsigned origDataLen)
{
	void *rtn = fmalloc(origDataLen);

	bcopy(origData, rtn, origDataLen);
	return rtn;
}

void ffree(void *data)
{
	#if		FALLOC_WATCH
	if(data == freeWatchAddrs) {
		printf("====ffree watchpoint (0x%x) hit\n",
			(unsigned)freeWatchAddrs);
	}
	#endif
	if(freeExt != NULL) {
		(freeExt)(data);
	}
	else {
		free(data);
	}
}

void *frealloc(void *oldPtr, unsigned newSize)
{
	#if		FALLOC_WATCH
	if(oldPtr == freeWatchAddrs) {
		printf("====frealloc watchpoint (0x%x) hit\n",
			(unsigned)freeWatchAddrs);
	}
	#endif
	if(reallocExt != NULL) {
		return (reallocExt)(oldPtr, newSize);
	}
	else {
		return realloc(oldPtr, newSize);
	}
}
