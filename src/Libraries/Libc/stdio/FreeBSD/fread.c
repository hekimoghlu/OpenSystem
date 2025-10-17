/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 1, 2023.
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
#if defined(LIBC_SCCS) && !defined(lint)
static char sccsid[] = "@(#)fread.c	8.2 (Berkeley) 12/11/93";
#endif /* LIBC_SCCS and not lint */
#include <sys/cdefs.h>
__FBSDID("$FreeBSD: src/lib/libc/stdio/fread.c,v 1.16 2009/07/12 13:09:43 ed Exp $");

#include "namespace.h"
#include <stdio.h>
#include <string.h>
#include "un-namespace.h"
#include "local.h"
#include "libc_private.h"
#include "libc_hooks_impl.h"

/*
 * MT-safe version
 */

size_t
fread(void * __restrict buf, size_t size, size_t count, FILE * __restrict fp)
{
	size_t ret;

	libc_hooks_will_write(fp, sizeof(*fp));

	FLOCKFILE(fp);
	ret = __fread(buf, size, count, fp);
	FUNLOCKFILE(fp);
	return (ret);
}

/*
 * The maximum amount to read to avoid integer overflow.  INT_MAX is odd,
 * so it make sense to make it even.  We subtract (BUFSIZ - 1) to get a
 * whole number of BUFSIZ chunks.
 */
#define MAXREAD	(INT_MAX - (BUFSIZ - 1))

/* __fread0: int sized, with size = 1 */
static inline int
__fread0(void * __restrict buf, int count, FILE * __restrict fp)
{
	int resid;
	char *p;
	int r, ret;

	resid = count;
	p = buf;
	/* first deal with anything left in buffer, plus any ungetc buffers */
	while (resid > (r = fp->_r)) {
		(void)memcpy((void *)p, (void *)fp->_p, (size_t)r);
		fp->_p += r;
		/* fp->_r = 0 ... done in __srefill */
		p += r;
		resid -= r;
		if ((ret = __srefill0(fp)) > 0)
			break;
		else if (ret) {
			/* no more input: return partial result */
			return (count - resid);
		}
	}
	/*
	 * 5980080: don't use optimization if __SMBF not set (meaning setvbuf
	 * was called, and the buffer belongs to the user).
	 * 6180417: but for unbuffered (__SMBF is not set), so specifically
	 * test for it.
	 */
	if ((fp->_flags & (__SMBF | __SNBF)) && resid > fp->_bf._size) {
		struct __sbuf save;
		size_t n;

		save = fp->_bf;
		fp->_bf._base = (unsigned char *)p;
		fp->_bf._size = resid;
		while (fp->_bf._size > 0) {
			if ((ret = __srefill1(fp)) != 0) {
				/* no more input: return partial result */
				resid = fp->_bf._size;
				fp->_bf = save;
				fp->_p = fp->_bf._base;
				/* fp->_r = 0;  already set in __srefill1 */
				return (count - resid);
			}
			fp->_bf._base += fp->_r;
			fp->_bf._size -= fp->_r;
		}
		fp->_bf = save;
		n = fp->_bf._size * ((resid - 1) / fp->_bf._size);
		r = resid - n;
		(void)memcpy((void *)fp->_bf._base, (void *)(p + n), (size_t)r);
		fp->_p = fp->_bf._base + r;
		fp->_r = 0;
	} else {
		while (resid > (r = fp->_r)) {
			(void)memcpy((void *)p, (void *)fp->_p, (size_t)r);
			fp->_p += r;
			/* fp->_r = 0 ... done in __srefill */
			p += r;
			resid -= r;
			if (__srefill1(fp)) {
				/* no more input: return partial result */
				return (count - resid);
			}
		}
		(void)memcpy((void *)p, (void *)fp->_p, resid);
		fp->_r -= resid;
		fp->_p += resid;
	}
	return (count);
}

size_t
__fread(void * __restrict buf, size_t size, size_t count, FILE * __restrict fp)
{
	size_t resid;
	int r, ret;
	size_t total;

	libc_hooks_will_write(buf, size * count);

	/*
	 * ANSI and SUSv2 require a return value of 0 if size or count are 0.
	 */
	if ((resid = count * size) == 0)
		return (0);
	ORIENT(fp, -1);
	if (fp->_r < 0)
		fp->_r = 0;

	for (total = resid; resid > 0; buf += r, resid -= r) {
		r = resid > INT_MAX ? MAXREAD : (int)resid;
		if ((ret = __fread0(buf, r, fp)) != r) {
			count = (total - resid + ret) / size;
			break;
		}
	}
	return (count);
}
