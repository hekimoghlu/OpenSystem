/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 27, 2022.
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
static char sccsid[] = "@(#)freopen.c	8.1 (Berkeley) 6/4/93";
#endif /* LIBC_SCCS and not lint */
#include <sys/cdefs.h>
__FBSDID("$FreeBSD: src/lib/libc/stdio/freopen.c,v 1.21 2008/04/17 22:17:54 jhb Exp $");

#include "namespace.h"
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <errno.h>
#include <limits.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include "un-namespace.h"
#include "libc_private.h"
#include "local.h"

/*
 * Re-direct an existing, open (probably) file to some other file.
 * ANSI is written such that the original file gets closed if at
 * all possible, no matter what.
 */
FILE *
freopen(file, mode, fp)
	const char * __restrict file;
	const char * __restrict mode;
	FILE *fp;
{
	int f;
	int dflags, flags, isopen, oflags, sverrno, wantfd;

	if ((flags = __sflags(mode, &oflags)) == 0) {
		sverrno = errno;
		(void) fclose(fp);
		errno = sverrno;
		return (NULL);
	}

	pthread_once(&__sdidinit, __sinit);
	
	FLOCKFILE(fp);

	/*
	 * If the filename is a NULL pointer, the caller is asking us to
	 * re-open the same file with a different mode. We allow this only
	 * if the modes are compatible.
	 */
	if (file == NULL) {
		/* See comment below regarding freopen() of closed files. */
		if (fp->_flags == 0) {
			FUNLOCKFILE(fp);
			errno = EINVAL;
			return (NULL);
		}
		if ((dflags = _fcntl(fp->_file, F_GETFL)) < 0) {
			sverrno = errno;
			fclose(fp);
			FUNLOCKFILE(fp);
			errno = sverrno;
			return (NULL);
		}
		if ((dflags & O_ACCMODE) != O_RDWR && (dflags & O_ACCMODE) !=
		    (oflags & O_ACCMODE)) {
			fclose(fp);
			FUNLOCKFILE(fp);
			errno = EBADF;
			return (NULL);
		}
		if (fp->_flags & __SWR)
			(void) __sflush(fp);
		if ((oflags ^ dflags) & O_APPEND) {
			dflags &= ~O_APPEND;
			dflags |= oflags & O_APPEND;
			if (_fcntl(fp->_file, F_SETFL, dflags) < 0) {
				sverrno = errno;
				fclose(fp);
				FUNLOCKFILE(fp);
				errno = sverrno;
				return (NULL);
			}
		}
		if (oflags & O_TRUNC)
			(void) ftruncate(fp->_file, (off_t)0);
		if (!(oflags & O_APPEND))
			(void) _sseek(fp, (fpos_t)0, SEEK_SET);
		f = fp->_file;
		isopen = 0;
		wantfd = -1;
		goto finish;
	}

	/*
	 * There are actually programs that depend on being able to "freopen"
	 * descriptors that weren't originally open.  Keep this from breaking.
	 * Remember whether the stream was open to begin with, and which file
	 * descriptor (if any) was associated with it.  If it was attached to
	 * a descriptor, defer closing it; freopen("/dev/stdin", "r", stdin)
	 * should work.  This is unnecessary if it was not a Unix file.
	 *
	 * For UNIX03, we always close if it was open.
	 */
	if (fp->_flags == 0) {
		fp->_flags = __SEOF;	/* hold on to it */
		isopen = 0;
		wantfd = -1;
	} else {
		/* flush the stream; ANSI doesn't require this. */
		if (fp->_flags & __SWR)
			(void) __sflush(fp);
		/* if close is NULL, closing is a no-op, hence pointless */
#if __DARWIN_UNIX03
		if (fp->_close)
			(void) (*fp->_close)(fp->_cookie);
		isopen = 0;
		wantfd = -1;
#else /* !__DARWIN_UNIX03 */
		isopen = fp->_close != NULL;
		if ((wantfd = fp->_file) < 0 && isopen) {
			(void) (*fp->_close)(fp->_cookie);
			isopen = 0;
		}
#endif /* __DARWIN_UNIX03 */
	}

	/* Get a new descriptor to refer to the new file. */
	f = _open(file, oflags, DEFFILEMODE);
	if (f < 0 && isopen) {
		/* If out of fd's close the old one and try again. */
		if (errno == ENFILE || errno == EMFILE) {
			(void) (*fp->_close)(fp->_cookie);
			isopen = 0;
			f = _open(file, oflags, DEFFILEMODE);
		}
	}
	sverrno = errno;

finish:
	/*
	 * Finish closing fp.  Even if the open succeeded above, we cannot
	 * keep fp->_base: it may be the wrong size.  This loses the effect
	 * of any setbuffer calls, but stdio has always done this before.
	 */
	if (isopen)
		(void) (*fp->_close)(fp->_cookie);
	if (fp->_flags & __SMBF)
		free((char *)fp->_bf._base);
	fp->_w = 0;
	fp->_r = 0;
	fp->_p = NULL;
	fp->_bf._base = NULL;
	fp->_bf._size = 0;
	fp->_lbfsize = 0;
	if (HASUB(fp))
		FREEUB(fp);
	fp->_ub._size = 0;
	if (HASLB(fp))
		FREELB(fp);
	fp->_lb._size = 0;
	fp->_orientation = 0;
	memset(&fp->_mbstate, 0, sizeof(mbstate_t));

	if (f < 0) {			/* did not get it after all */
		FUNLOCKFILE(fp);
		__sfprelease(fp);	/* set it free */
		errno = sverrno;	/* restore in case _close clobbered */
		return (NULL);
	}

	/*
	 * If reopening something that was open before on a real file, try
	 * to maintain the descriptor.  Various C library routines (perror)
	 * assume stderr is always fd STDERR_FILENO, even if being freopen'd.
	 */
	if (wantfd >= 0 && f != wantfd) {
		if (_dup2(f, wantfd) >= 0) {
			(void)_close(f);
			f = wantfd;
		}
	}

	/*
	 * File descriptors are a full int, but _file is only a short.
	 * If we get a valid file descriptor that is greater than
	 * SHRT_MAX, then the fd will get sign-extended into an
	 * invalid file descriptor.  Handle this case by failing the
	 * open.
	 */
	if (f > SHRT_MAX) {
		FUNLOCKFILE(fp);
		__sfprelease(fp);	/* set it free */
		errno = EMFILE;
		return (NULL);
	}

	fp->_flags = flags;
	fp->_file = f;
	fp->_cookie = fp;
	fp->_read = __sread;
	fp->_write = __swrite;
	fp->_seek = __sseek;
	fp->_close = __sclose;
	/*
	 * When opening in append mode, even though we use O_APPEND,
	 * we need to seek to the end so that ftell() gets the right
	 * answer.  If the user then alters the seek pointer, or
	 * the file extends, this will fail, but there is not much
	 * we can do about this.  (We could set __SAPP and check in
	 * fseek and ftell.)
	 */
	if (oflags & O_APPEND)
		(void) _sseek(fp, (fpos_t)0, SEEK_END);
	FUNLOCKFILE(fp);
	return (fp);
}
