/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 27, 2022.
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
static char sccsid[] = "@(#)fflush.c	8.1 (Berkeley) 6/4/93";
#endif /* LIBC_SCCS and not lint */
#include <sys/cdefs.h>
__FBSDID("$FreeBSD: src/lib/libc/stdio/fflush.c,v 1.14 2007/01/09 00:28:06 imp Exp $");

#include "namespace.h"
#include <errno.h>
#include <stdio.h>
#include <string.h>
#include "un-namespace.h"
#include "libc_private.h"
#include "local.h"
#include "libc_hooks_impl.h"

static int	sflush_locked(FILE *);

/*
 * Flush a single file, or (if fp is NULL) all files.
 * MT-safe version
 */
int
fflush(FILE *fp)
{
	int retval = 0;

	if (fp == NULL) {
		return (_fwalk(sflush_locked));
	}

	libc_hooks_will_write(fp, sizeof(*fp));

	FLOCKFILE(fp);
	retval = __sflush(fp);
	FUNLOCKFILE(fp);
	return (retval);
}

/*
 * Flush a single file, or (if fp is NULL) all files.
 * Non-MT-safe version
 */
int
__fflush(FILE *fp)
{
	int retval;

	if (fp == NULL)
		return (_fwalk(sflush_locked));
	if ((fp->_flags & (__SWR | __SRW)) == 0) {
		retval = 0;
	} else
		retval = __sflush(fp);
	return (retval);
}

int
__sflush(FILE *fp)
{
	unsigned char *p;
	int f, n, t;

	f = fp->_flags;

	if ((p = fp->_bf._base) == NULL)
		return (0);

	/*
	 * SUSv3 requires that fflush() on a seekable input stream updates the file
	 * position indicator with the underlying seek function.  Use a dumb fseek
	 * for this (don't attempt to preserve the buffers).
	 */
	if ((f & __SRD) != 0) {
		if (fp->_seek == NULL) {
			/*
			 * No way to seek this file -- just return "success."
			 */
			return (0);
		}

		n = fp->_r;

		if (n > 0) {
			/*
			 * See _fseeko's dumb path.
			 */
			if (_sseek(fp, (fpos_t)-n, SEEK_CUR) == -1) {
				if (errno == ESPIPE) {
					/*
					 * Ignore ESPIPE errors, since there's no way to put the bytes
					 * back into the pipe.
					 */
					return (0);
				}
				return (EOF);
			}

			if (HASUB(fp)) {
				FREEUB(fp);
			}
			fp->_p = fp->_bf._base;
			fp->_r = 0;
			fp->_flags &= ~__SEOF;
			memset(&fp->_mbstate, 0, sizeof(mbstate_t));
		}
		return (0);
	}

	if ((f & __SWR) != 0) {
		n = fp->_p - p;		/* write this much */

		/*
		 * Set these immediately to avoid problems with longjmp and to allow
		 * exchange buffering (via setvbuf) in user write function.
		 */
		fp->_p = p;
		fp->_w = f & (__SLBF|__SNBF) ? 0 : fp->_bf._size;

		for (; n > 0; n -= t, p += t) {
			t = _swrite(fp, (char *)p, n);
			if (t <= 0) {
				if (p > fp->_p)
					/* some was written */
					memmove(fp->_p, p, n);
				fp->_p += n;
				if ((fp->_flags & __SNBF) == 0)
					fp->_w -= n;
				fp->_flags |= __SERR;
				return (EOF);
			}
		}
	}
	return (0);
}

static int
sflush_locked(FILE *fp)
{
	int	ret;

	FLOCKFILE(fp);
	ret = __sflush(fp);
	FUNLOCKFILE(fp);
	return (ret);
}
