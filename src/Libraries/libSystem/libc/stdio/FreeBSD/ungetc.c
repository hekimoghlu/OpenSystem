/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 5, 2022.
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
static char sccsid[] = "@(#)ungetc.c	8.2 (Berkeley) 11/3/93";
#endif /* LIBC_SCCS and not lint */
#include <sys/cdefs.h>
__FBSDID("$FreeBSD: src/lib/libc/stdio/ungetc.c,v 1.18 2008/04/17 22:17:54 jhb Exp $");

#include "namespace.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "un-namespace.h"
#include "local.h"
#include "libc_private.h"

static int __submore(FILE *);

/*
 * Expand the ungetc buffer `in place'.  That is, adjust fp->_p when
 * the buffer moves, so that it points the same distance from the end,
 * and move the bytes in the buffer around as necessary so that they
 * are all at the end (stack-style).
 */
static int
__submore(FILE *fp)
{
	int i;
	unsigned char *p;

	if (fp->_ub._base == fp->_ubuf) {
		/*
		 * Get a new buffer (rather than expanding the old one).
		 */
		if ((p = malloc((size_t)BUFSIZ)) == NULL)
			return (EOF);
		fp->_ub._base = p;
		fp->_ub._size = BUFSIZ;
		p += BUFSIZ - sizeof(fp->_ubuf);
		for (i = sizeof(fp->_ubuf); --i >= 0;)
			p[i] = fp->_ubuf[i];
		fp->_p = p;
		return (0);
	}
	i = fp->_ub._size;
	p = realloc(fp->_ub._base, (size_t)(i << 1));
	if (p == NULL)
		return (EOF);
	/* no overlap (hence can use memcpy) because we doubled the size */
	(void)memcpy((void *)(p + i), (void *)p, (size_t)i);
	fp->_p = p + i;
	fp->_ub._base = p;
	fp->_ub._size = i << 1;
	return (0);
}

/*
 * MT-safe version
 */
int
ungetc(int c, FILE *fp)
{
	int ret;

	pthread_once(&__sdidinit, __sinit);

	FLOCKFILE(fp);
	ORIENT(fp, -1);
	ret = __ungetc(c, fp);
	FUNLOCKFILE(fp);
	return (ret);
}

/*
 * Non-MT-safe version
 */
int
__ungetc(int c, FILE *fp)
{

	if (c == EOF)
		return (EOF);
	if ((fp->_flags & __SRD) == 0) {
		/*
		 * Not already reading: no good unless reading-and-writing.
		 * Otherwise, flush any current write stuff.
		 */
		if ((fp->_flags & __SRW) == 0)
			return (EOF);
		if (fp->_flags & __SWR) {
			if (__sflush(fp))
				return (EOF);
			fp->_flags &= ~__SWR;
			fp->_w = 0;
			fp->_lbfsize = 0;
		}
		fp->_flags |= __SRD;
	}
	c = (unsigned char)c;

	/*
	 * If we are in the middle of ungetc'ing, just continue.
	 * This may require expanding the current ungetc buffer.
	 */
	if (HASUB(fp)) {
		if (fp->_r >= fp->_ub._size && __submore(fp))
			return (EOF);
		*--fp->_p = c;
		fp->_r++;
		return (c);
	}
	fp->_flags &= ~__SEOF;

	/*
	 * If we can handle this by simply backing up, do so,
	 * but never replace the original character.
	 * (This makes sscanf() work when scanning `const' data.)
	 */
	if (fp->_bf._base != NULL && fp->_p > fp->_bf._base &&
	    fp->_p[-1] == c) {
		fp->_p--;
		fp->_r++;
		return (c);
	}

	/*
	 * Create an ungetc buffer.
	 * Initially, we will use the `reserve' buffer.
	 */
	fp->_ur = fp->_r;
	fp->_up = fp->_p;
	fp->_ub._base = fp->_ubuf;
	fp->_ub._size = sizeof(fp->_ubuf);
	fp->_ubuf[sizeof(fp->_ubuf) - 1] = c;
	fp->_p = &fp->_ubuf[sizeof(fp->_ubuf) - 1];
	fp->_r = 1;
	return (c);
}
