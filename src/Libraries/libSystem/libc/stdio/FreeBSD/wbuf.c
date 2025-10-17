/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 6, 2023.
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
static char sccsid[] = "@(#)wbuf.c	8.1 (Berkeley) 6/4/93";
#endif /* LIBC_SCCS and not lint */
#include <sys/cdefs.h>
__FBSDID("$FreeBSD: src/lib/libc/stdio/wbuf.c,v 1.12 2007/01/09 00:28:08 imp Exp $");

#include <stdio.h>
#include <errno.h>
#include "local.h"

/*
 * Write the given character into the (probably full) buffer for
 * the given file.  Flush the buffer out if it is or becomes full,
 * or if c=='\n' and the file is line buffered.
 *
 * Non-MT-safe
 */
int
__swbuf(c, fp)
	int c;
	FILE *fp;
{
	int n;

	/*
	 * In case we cannot write, or longjmp takes us out early,
	 * make sure _w is 0 (if fully- or un-buffered) or -_bf._size
	 * (if line buffered) so that we will get called again.
	 * If we did not do this, a sufficient number of putc()
	 * calls might wrap _w from negative to positive.
	 */
	fp->_w = fp->_lbfsize;
	if (prepwrite(fp) != 0) {
		errno = EBADF;
		return (EOF);
	}
	c = (unsigned char)c;

	ORIENT(fp, -1);

	/*
	 * If it is completely full, flush it out.  Then, in any case,
	 * stuff c into the buffer.  If this causes the buffer to fill
	 * completely, or if c is '\n' and the file is line buffered,
	 * flush it (perhaps a second time).  The second flush will always
	 * happen on unbuffered streams, where _bf._size==1; fflush()
	 * guarantees that putc() will always call wbuf() by setting _w
	 * to 0, so we need not do anything else.
	 */
	n = fp->_p - fp->_bf._base;
	if (n >= fp->_bf._size) {
		if (__fflush(fp))
			return (EOF);
		n = 0;
	}
	fp->_w--;
	*fp->_p++ = c;
	if (++n == fp->_bf._size || (fp->_flags & __SLBF && c == '\n'))
		if (__fflush(fp))
			return (EOF);
	return (c);
}
