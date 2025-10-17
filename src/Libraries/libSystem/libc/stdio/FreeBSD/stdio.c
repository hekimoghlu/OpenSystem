/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 30, 2025.
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
static char sccsid[] = "@(#)stdio.c	8.1 (Berkeley) 6/4/93";
#endif /* LIBC_SCCS and not lint */
#include <sys/cdefs.h>
__FBSDID("$FreeBSD: src/lib/libc/stdio/stdio.c,v 1.28 2008/05/05 16:14:02 jhb Exp $");

#include "namespace.h"
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include "un-namespace.h"
#include "local.h"

/*
 * Small standard I/O/seek/close functions.
 */
int
__sread(cookie, buf, n)
	void *cookie;
	char *buf;
	int n;
{
	FILE *fp = cookie;

	return(_read(fp->_file, buf, (size_t)n));
}

int
__swrite(cookie, buf, n)
	void *cookie;
	char const *buf;
	int n;
{
	FILE *fp = cookie;

	return (_write(fp->_file, buf, (size_t)n));
}

fpos_t
__sseek(cookie, offset, whence)
	void *cookie;
	fpos_t offset;
	int whence;
{
	FILE *fp = cookie;

	return (lseek(fp->_file, (off_t)offset, whence));
}

int
__sclose(cookie)
	void *cookie;
{

	return (_close(((FILE *)cookie)->_file));
}

/*
 * Higher level wrappers.
 */
int
_sread(fp, buf, n)
	FILE *fp;
	char *buf;
	int n;
{
	int ret;

	ret = (*fp->_read)(fp->_cookie, buf, n);
	if (ret > 0) {
		if (fp->_flags & __SOFF) {
			if (fp->_offset <= OFF_MAX - ret)
				fp->_offset += ret;
			else
				fp->_flags &= ~__SOFF;
		}
	} else if (ret < 0)
		fp->_flags &= ~__SOFF;
	return (ret);
}

int
_swrite(fp, buf, n)
	FILE *fp;
	char const *buf;
	int n;
{
	int ret;
	int serrno;

	if (fp->_flags & __SAPP) {
		serrno = errno;
		if (_sseek(fp, (fpos_t)0, SEEK_END) == -1 &&
		    (fp->_flags & __SOPT))
			return (-1);
		errno = serrno;
	}
	ret = (*fp->_write)(fp->_cookie, buf, n);
	/* __SOFF removed even on success in case O_APPEND mode is set. */
	if (ret >= 0) {
		if ((fp->_flags & (__SAPP|__SOFF)) == (__SAPP|__SOFF) &&
		    fp->_offset <= OFF_MAX - ret)
			fp->_offset += ret;
		else
			fp->_flags &= ~__SOFF;

	} else if (ret < 0)
		fp->_flags &= ~__SOFF;
	return (ret);
}

fpos_t
_sseek(fp, offset, whence)
	FILE *fp;
	fpos_t offset;
	int whence;
{
	fpos_t ret;
	int serrno, errret;

	serrno = errno;
	errno = 0;
	ret = (*fp->_seek)(fp->_cookie, offset, whence);
	errret = errno;
	if (errno == 0)
		errno = serrno;
	/*
	 * Disallow negative seeks per POSIX.
	 * It is needed here to help upper level caller
	 * in the cases it can't detect.
	 */
	if (ret < 0) {
		if (errret == 0) {
			if (offset != 0 || whence != SEEK_CUR) {
				if (HASUB(fp))
					FREEUB(fp);
				fp->_p = fp->_bf._base;
				fp->_r = 0;
				fp->_flags &= ~__SEOF;
			}
			fp->_flags |= __SERR;
			errno = EINVAL;
		} else if (errret == ESPIPE)
			fp->_flags &= ~__SAPP;
		fp->_flags &= ~__SOFF;
		ret = -1;
	} else if (fp->_flags & __SOPT) {
		fp->_flags |= __SOFF;
		fp->_offset = ret;
	}
	return (ret);
}
