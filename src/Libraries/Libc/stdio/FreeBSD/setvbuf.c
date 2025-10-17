/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 9, 2024.
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
static char sccsid[] = "@(#)setvbuf.c	8.2 (Berkeley) 11/16/93";
#endif /* LIBC_SCCS and not lint */
#include <sys/cdefs.h>
__FBSDID("$FreeBSD: src/lib/libc/stdio/setvbuf.c,v 1.14 2007/01/09 00:28:07 imp Exp $");

#include "namespace.h"
#include <stdio.h>
#include <stdlib.h>
#include "un-namespace.h"
#include "local.h"
#include "libc_private.h"
#include "libc_hooks_impl.h"

/*
 * Set one of the three kinds of buffering, optionally including
 * a buffer.
 */
int
setvbuf(FILE * __restrict fp, char * __restrict buf, int mode, size_t size)
{
	int ret, flags;
	size_t iosize;
	int ttyflag;

	/*
	 * Verify arguments.  The `int' limit on `size' is due to this
	 * particular implementation.  Note, buf and size are ignored
	 * when setting _IONBF.
	 */
	if (mode != _IONBF)
		if ((mode != _IOFBF && mode != _IOLBF) || (int)size < 0)
			return (EOF);

	libc_hooks_will_write(fp, sizeof(*fp));
	libc_hooks_will_write(buf, size);

	FLOCKFILE(fp);
	/*
	 * Write current buffer, if any.  Discard unread input (including
	 * ungetc data), cancel line buffering, and free old buffer if
	 * malloc()ed.  We also clear any eof condition, as if this were
	 * a seek.
	 */
	ret = 0;
	(void)__sflush(fp);
	if (HASUB(fp))
		FREEUB(fp);
	fp->_r = fp->_lbfsize = 0;
	flags = fp->_flags;
	if (flags & __SMBF)
		free((void *)fp->_bf._base);
	flags &= ~(__SLBF | __SNBF | __SMBF | __SOPT | __SOFF | __SNPT | __SEOF);

	/* If setting unbuffered mode, skip all the hard work. */
	if (mode == _IONBF)
		goto nbf;

	/*
	 * Find optimal I/O size for seek optimization.  This also returns
	 * a `tty flag' to suggest that we check isatty(fd), but we do not
	 * care since our caller told us how to buffer.
	 */
	flags |= __swhatbuf(fp, &iosize, &ttyflag);
	if (size == 0) {
		buf = NULL;	/* force local allocation */
		size = iosize;
	}

	/* Allocate buffer if needed. */
	if (buf == NULL) {
		if ((buf = malloc(size)) == NULL) {
			/*
			 * Unable to honor user's request.  We will return
			 * failure, but try again with file system size.
			 */
			ret = EOF;
			if (size != iosize) {
				size = iosize;
				buf = malloc(size);
			}
		}
		if (buf == NULL) {
			/* No luck; switch to unbuffered I/O. */
nbf:
			fp->_flags = flags | __SNBF;
			fp->_w = 0;
			fp->_bf._base = fp->_p = fp->_nbuf;
			fp->_bf._size = 1;
			FUNLOCKFILE(fp);
			return (ret);
		}
		flags |= __SMBF;
	}

	/*
	 * Kill any seek optimization if the buffer is not the
	 * right size.
	 *
	 * SHOULD WE ALLOW MULTIPLES HERE (i.e., ok iff (size % iosize) == 0)?
	 */
	if (size != iosize)
		flags |= __SNPT;

	/*
	 * Fix up the FILE fields, and set __cleanup for output flush on
	 * exit (since we are buffered in some way).
	 */
	if (mode == _IOLBF)
		flags |= __SLBF;
	fp->_flags = flags;
	fp->_bf._base = fp->_p = (unsigned char *)buf;
	fp->_bf._size = size;
	/* fp->_lbfsize is still 0 */
	if (flags & __SWR) {
		/*
		 * Begin or continue writing: see __swsetup().  Note
		 * that __SNBF is impossible (it was handled earlier).
		 */
		if (flags & __SLBF) {
			fp->_w = 0;
			fp->_lbfsize = -fp->_bf._size;
		} else
			fp->_w = size;
	} else {
		/* begin/continue reading, or stay in intermediate state */
		fp->_w = 0;
	}
#ifdef __APPLE__
	__cleanup = 1;
#else
	__cleanup = _cleanup;
#endif // __APPLE__

	FUNLOCKFILE(fp);
	return (ret);
}
