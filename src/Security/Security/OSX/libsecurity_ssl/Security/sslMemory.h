/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 27, 2024.
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
 * sslMemory.h - SSLBuffer and Memory allocator declarations
 */

/* This header should be kernel safe */

#ifndef _SSLMEMORY_H_
#define _SSLMEMORY_H_ 1

#include "sslTypes.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * General purpose allocators
 */
void *sslMalloc(size_t length);
void sslFree(void *p);

/*
 * SSLBuffer-oriented allocators
 */
int SSLAllocBuffer(SSLBuffer *buf, size_t length);
int SSLFreeBuffer(SSLBuffer *buf);

/*
 * Convenience routines
 */
uint8_t *sslAllocCopy(const uint8_t *src, size_t len);
int SSLCopyBufferFromData(
	const void *src,
	size_t len,
	SSLBuffer *dst);		// data mallocd and returned 
int SSLCopyBuffer(
	const SSLBuffer *src, 
	SSLBuffer *dst);		// data mallocd and returned

#ifdef __cplusplus
}
#endif

#define SET_SSL_BUFFER(buf, d, l)   do { (buf).data = (d); (buf).length = (l); } while (0)

#endif
