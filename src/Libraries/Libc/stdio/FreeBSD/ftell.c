/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 22, 2023.
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
static char sccsid[] = "@(#)ftell.c	8.2 (Berkeley) 5/4/95";
#endif /* LIBC_SCCS and not lint */
#include <sys/cdefs.h>
__FBSDID("$FreeBSD$");

#include "namespace.h"
#include <sys/types.h>
#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include "un-namespace.h"
#include "local.h"
#include "libc_private.h"
#include "../stdio_init.h"

/*
 * standard ftell function.
 */
long
ftell(FILE *fp)
{
	off_t rv;

	rv = ftello(fp);
	if (rv > LONG_MAX) {
		errno = EOVERFLOW;
		return (-1);
	}
	return (rv);
}

/*
 * ftello: return current offset.
 */
off_t
ftello(FILE *fp)
{
	fpos_t rv;
	int ret;

	FLOCKFILE(fp);
	ret = _ftello(fp, &rv);
	FUNLOCKFILE(fp);
	if (ret)
		return (-1);
	if (rv < 0) {   /* Unspecified value because of ungetc() at 0 */
		errno = ESPIPE;
		return (-1);
	}
	return (rv);
}

int
_ftello(FILE *fp, fpos_t *offset)
{
	fpos_t pos;
	size_t n;

	if (fp->_seek == NULL) {
		errno = ESPIPE;			/* historic practice */
		return (1);
	}

	/*
	 * Find offset of underlying I/O object, then
	 * adjust for buffered bytes.
	 */

	/* Conformance fix not applied for pre-macOS 13.0 binaries. (96211868) */
	if (__ftell_conformance_fix) {
		if (!(fp->_flags & __SRD) && (fp->_flags & __SWR) &&
			fp->_p != NULL && fp->_p - fp->_bf._base > 0 &&
			((fp->_flags & __SAPP))) {
			pos = _sseek(fp, (fpos_t)0, SEEK_END);
			if (pos == -1)
				return (1);
		} else if (fp->_flags & __SOFF) {
			pos = fp->_offset;
		} else {
			pos = _sseek(fp, (fpos_t)0, SEEK_CUR);
			if (pos == -1)
				return (1);
		}
	} else {
		if (__sflush(fp))		/* may adjust seek offset on append stream */
			return (1);

		if (fp->_flags & __SOFF) {
			pos = fp->_offset;
		} else {
			pos = _sseek(fp, (fpos_t)0, SEEK_CUR);
			if (pos == -1)
				return (1);
		}
	}
	
	if (fp->_flags & __SRD) {
		/*
		 * Reading.  Any unread characters (including
		 * those from ungetc) cause the position to be
		 * smaller than that in the underlying object.
		 */
		if ((pos -= (HASUB(fp) ? fp->_ur : fp->_r)) < 0) {
			fp->_flags |= __SERR;
			errno = EIO;
			return (1);
		}
		if (HASUB(fp))
			pos -= fp->_r;  /* Can be negative at this point. */
	} else if ((fp->_flags & __SWR) && fp->_p != NULL) {
		/*
		 * Writing.  Any buffered characters cause the
		 * position to be greater than that in the
		 * underlying object.
		 */
		n = fp->_p - fp->_bf._base;
		if (__ftell_conformance_fix && n <= 0) {
			goto out;
		}

		if (pos > OFF_MAX - n) {
			errno = EOVERFLOW;
			return (1);
		}
		pos += n;
	}

out:
	*offset = pos;
	return (0);
}
