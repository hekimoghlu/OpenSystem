/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 5, 2025.
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
static char sccsid[] = "@(#)fclose.c	8.1 (Berkeley) 6/4/93";
#endif /* LIBC_SCCS and not lint */
#include <sys/cdefs.h>
__FBSDID("$FreeBSD: src/lib/libc/stdio/fclose.c,v 1.12 2007/01/09 00:28:06 imp Exp $");

#include "namespace.h"
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include "un-namespace.h"
#include "libc_private.h"
#include "local.h"

int
fclose(FILE *fp)
{
	int r;
	int error = 0;

	pthread_once(&__sdidinit, __sinit);

	if (fp == NULL) {
		errno = EFAULT;
		return (EOF);
	}
	if (fp->_flags == 0) {	/* not open! */
		errno = EBADF;
		return (EOF);
	}
	FLOCKFILE(fp);
	r = __sflush(fp);
	if (r < 0) {
		error = errno;
	}
	if (fp->_close != NULL && (*fp->_close)(fp->_cookie) < 0) {
		r = EOF;
		error = errno;
	}
	if (fp->_flags & __SMBF)
		free((char *)fp->_bf._base);
	if (HASUB(fp))
		FREEUB(fp);
	if (HASLB(fp))
		FREELB(fp);
	fp->_file = -1;
	fp->_r = fp->_w = 0;	/* Mess up if reaccessed. */
	FUNLOCKFILE(fp);
	__sfprelease(fp);	/* Release this FILE for reuse. */
	/* Don't clobber errno unnecessarily. */
	if (error) {
		errno = error;
	}
	return (r);
}
