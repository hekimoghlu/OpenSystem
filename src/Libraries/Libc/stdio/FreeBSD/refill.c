/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 19, 2021.
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
static char sccsid[] = "@(#)refill.c	8.1 (Berkeley) 6/4/93";
#endif /* LIBC_SCCS and not lint */
#include <sys/cdefs.h>
__FBSDID("$FreeBSD: src/lib/libc/stdio/refill.c,v 1.20 2008/04/17 22:17:54 jhb Exp $");

#include "namespace.h"
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include "un-namespace.h"

#include "libc_private.h"
#include "local.h"

static int lflush(FILE *);

static int
lflush(FILE *fp)
{
	int	ret = 0;

	if ((fp->_flags & (__SLBF|__SWR)) == (__SLBF|__SWR)) {
		FLOCKFILE(fp);
		ret = __sflush(fp);
		FUNLOCKFILE(fp);
	}
	return (ret);
}

/*
 * Refill a stdio buffer.
 * Return EOF on eof or error, 0 otherwise.
 */
__private_extern__ int
__srefill0(FILE *fp)
{

	/* make sure stdio is set up */
	pthread_once(&__sdidinit, __sinit);

	ORIENT(fp, -1);

	fp->_r = 0;		/* largely a convenience for callers */

	/* SysV does not make this test; take it out for compatibility */
	if (fp->_flags & __SEOF)
		return (EOF);

	/* if not already reading, have to be reading and writing */
	if ((fp->_flags & __SRD) == 0) {
		if ((fp->_flags & __SRW) == 0) {
			errno = EBADF;
			fp->_flags |= __SERR;
			return (EOF);
		}
		/* switch to reading */
		if (fp->_flags & __SWR) {
			if (__sflush(fp))
				return (EOF);
			fp->_flags &= ~__SWR;
			fp->_w = 0;
			fp->_lbfsize = 0;
		}
		fp->_flags |= __SRD;
	} else {
		/*
		 * We were reading.  If there is an ungetc buffer,
		 * we must have been reading from that.  Drop it,
		 * restoring the previous buffer (if any).  If there
		 * is anything in that buffer, return.
		 */
		if (HASUB(fp)) {
			FREEUB(fp);
			if ((fp->_r = fp->_ur) != 0) {
				fp->_p = fp->_up;
				return (0);
			}
		}
	}

	if (fp->_bf._base == NULL)
		__smakebuf(fp);

	/*
	 * Before reading from a line buffered or unbuffered file,
	 * flush all line buffered output files, per the ANSI C
	 * standard.
	 */
	if (fp->_flags & (__SLBF|__SNBF)) {
		/* Ignore this file in _fwalk to avoid potential deadlock. */
		fp->_flags |= __SIGN;
		(void) _fwalk(lflush);
		fp->_flags &= ~__SIGN;

		/* Now flush this file without locking it. */
		if ((fp->_flags & (__SLBF|__SWR)) == (__SLBF|__SWR))
			__sflush(fp);
	}
	return (1);
}

__private_extern__ int
__srefill1(FILE *fp)
{

	fp->_p = fp->_bf._base;
	fp->_r = _sread(fp, (char *)fp->_p, fp->_bf._size);
	fp->_flags &= ~__SMOD;	/* buffer contents are again pristine */
	if (fp->_r <= 0) {
		if (fp->_r == 0)
			fp->_flags |= __SEOF;
		else {
			fp->_r = 0;
			fp->_flags |= __SERR;
		}
		return (EOF);
	}
	return (0);
}

int
__srefill(FILE *fp)
{
	int ret;

	if ((ret = __srefill0(fp)) <= 0)
		return ret;
	return __srefill1(fp);
}
