/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 22, 2022.
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
 * sslMemory.c - Memory allocator implementation
 */

/* THIS FILE CONTAINS KERNEL CODE */

#include "sslMemory.h"
#include "sslDebug.h"

#include <string.h>			/* memset */
#include <AssertMacros.h>

// MARK: -
// MARK: Basic low-level malloc/free

/*
 * For now, all allocs/frees go thru here.
 */

#ifdef KERNEL

/* BSD Malloc */
#include <sys/malloc.h>
#include <IOKit/IOLib.h>
#include <libkern/libkern.h>

/* Define this for debugging sslMalloc and sslFree */
//#define SSL_CANARIS

void *
sslMalloc(size_t length)
{
    void *p;

#ifdef SSL_CANARIS
    length+=8;
#endif
    
    p = _MALLOC(length, M_TEMP, M_WAITOK);
    check(p);
    
    if(p==NULL)
        return p;
    
#ifdef SSL_CANARIS
    *(uint32_t *)p=(uint32_t)length-8;
    printf("sslMalloc @%p of 0x%08lx bytes\n", p, length-8);
    *(uint32_t *)(p+length-4)=0xdeadbeed;
    p+=4;
#endif

    return p;
}

void
sslFree(void *p)
{
	if(p != NULL) {

#ifdef SSL_CANARIS
        p=p-4;
        uint32_t len=*(uint32_t *)p;
        uint32_t marker=*(uint32_t *)(p+4+len);
        printf("sslFree @%p len=0x%08x\n", p, len);
        if(marker!=0xdeadbeef)
            panic("Buffer overflow in SSL!\n");
#endif
        
        _FREE(p, M_TEMP);
	}
}

#else

#include <stdlib.h>

void *
sslMalloc(size_t length)
{
	return malloc(length);
}

void
sslFree(void *p)
{   
	if(p != NULL) {
		free(p);
	}
}

#endif

// MARK: -
// MARK: SSLBuffer-level alloc/free

int SSLAllocBuffer(
	SSLBuffer *buf,
	size_t length)
{
	buf->data = (uint8_t *)sslMalloc(length);
	if(buf->data == NULL) {
        sslErrorLog("SSLAllocBuffer: NULL buf!\n");
        check(0);
		buf->length = 0;
		return -1;
	}
    buf->length = length;
    return 0;
}

int
SSLFreeBuffer(SSLBuffer *buf)
{   
	if(buf == NULL) {
		sslErrorLog("SSLFreeBuffer: NULL buf!\n");
        check(0);
		return -1;
	}
    sslFree(buf->data);
    buf->data = NULL;
    buf->length = 0;
    return 0;
}

uint8_t *sslAllocCopy(
	const uint8_t *src,
	size_t len)
{
	uint8_t *dst;
	
	dst = (uint8_t *)sslMalloc(len);
	if(dst == NULL) {
		return NULL;
	}
	memmove(dst, src, len);
	return dst;
}

int SSLCopyBufferFromData(
	const void *src,
	size_t len,
	SSLBuffer *dst)		// data mallocd and returned 
{   
	dst->data = sslAllocCopy((const uint8_t *)src, len);
	if(dst->data == NULL) {
        sslErrorLog("SSLCopyBufferFromData: NULL buf!\n");
        check(0);
		return -1;
	}
    dst->length = len;
    return 0;
}

int SSLCopyBuffer(
	const SSLBuffer *src, 
	SSLBuffer *dst)		// data mallocd and returned 
{   
	return SSLCopyBufferFromData(src->data, src->length, dst);
}

