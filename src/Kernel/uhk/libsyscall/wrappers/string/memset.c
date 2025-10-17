/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 11, 2022.
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
#include "strings.h"
#include <sys/types.h>

#define wsize   sizeof(u_int)
#define wmask   (wsize - 1)

// n.b. this must be compiled with -fno-builtin or it might get optimized into
// a recursive call to bzero.
__attribute__((visibility("hidden")))
void
_libkernel_bzero(void *dst0, size_t length)
{
	return (void)_libkernel_memset(dst0, 0, length);
}

#define RETURN  return (dst0)
#define VAL     c0
#define WIDEVAL c

__attribute__((visibility("hidden")))
void *
_libkernel_memset(void *dst0, int c0, size_t length)
{
	size_t t;
	u_int c;
	u_char *dst;

	dst = dst0;
	/*
	 * If not enough words, just fill bytes.  A length >= 2 words
	 * guarantees that at least one of them is `complete' after
	 * any necessary alignment.  For instance:
	 *
	 *	|-----------|-----------|-----------|
	 *	|00|01|02|03|04|05|06|07|08|09|0A|00|
	 *	          ^---------------------^
	 *		 dst		 dst+length-1
	 *
	 * but we use a minimum of 3 here since the overhead of the code
	 * to do word writes is substantial.
	 */
	if (length < 3 * wsize) {
		while (length != 0) {
			*dst++ = VAL;
			--length;
		}
		RETURN;
	}

	if ((c = (u_char)c0) != 0) {    /* Fill the word. */
		c = (c << 8) | c;       /* u_int is 16 bits. */
#if UINT_MAX > 0xffff
		c = (c << 16) | c;      /* u_int is 32 bits. */
#endif
#if UINT_MAX > 0xffffffff
		c = (c << 32) | c;      /* u_int is 64 bits. */
#endif
	}
	/* Align destination by filling in bytes. */
	if ((t = (long)dst & wmask) != 0) {
		t = wsize - t;
		length -= t;
		do {
			*dst++ = VAL;
		} while (--t != 0);
	}

	/* Fill words.  Length was >= 2*words so we know t >= 1 here. */
	t = length / wsize;
	do {
		*(u_int *)dst = WIDEVAL;
		dst += wsize;
	} while (--t != 0);

	/* Mop up trailing bytes, if any. */
	t = length & wmask;
	if (t != 0) {
		do {
			*dst++ = VAL;
		} while (--t != 0);
	}
	RETURN;
}
