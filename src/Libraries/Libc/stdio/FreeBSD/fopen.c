/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 11, 2025.
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
#ifdef VARIANT_DARWINEXTSN
#define _DARWIN_UNLIMITED_STREAMS
#define COUNT	0
#elif defined(VARIANT_LEGACY)
#define COUNT	0
#else
#define COUNT	1
#endif

#if defined(LIBC_SCCS) && !defined(lint)
static char sccsid[] = "@(#)fopen.c	8.1 (Berkeley) 6/4/93";
#endif /* LIBC_SCCS and not lint */
#include <sys/cdefs.h>
__FBSDID("$FreeBSD: src/lib/libc/stdio/fopen.c,v 1.14 2008/04/22 17:03:32 jhb Exp $");

#include "namespace.h"
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdio.h>
#include <errno.h>
#include <limits.h>
#include "un-namespace.h"

#include "local.h"
#include "libc_hooks_impl.h"

FILE *
fopen(const char * __restrict file, const char * __restrict mode)
{
	FILE *fp;
	int f;
	int flags, oflags;

	libc_hooks_will_read_cstring(file);
	libc_hooks_will_read_cstring(mode);

	if ((flags = __sflags(mode, &oflags)) == 0)
		return (NULL);
	if ((fp = __sfp(COUNT)) == NULL)
		return (NULL);
	if ((f = _open(file, oflags, DEFFILEMODE)) < 0) {
		__sfprelease(fp);		/* release */
		return (NULL);
	}
	/*
	 * File descriptors are a full int, but _file is only a short.
	 * If we get a valid file descriptor that is greater than
	 * SHRT_MAX, then the fd will get sign-extended into an
	 * invalid file descriptor.  Handle this case by failing the
	 * open.
	 */
	if (f > SHRT_MAX) {
		fp->_flags = 0;			/* release */
		_close(f);
		errno = EMFILE;
		return (NULL);
	}
	fp->_file = f;
	fp->_flags = flags;
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
		(void)_sseek(fp, (fpos_t)0, SEEK_END);
	return (fp);
}
