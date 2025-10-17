/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 12, 2025.
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
static char sccsid[] = "@(#)flags.c	8.1 (Berkeley) 6/4/93";
#endif /* LIBC_SCCS and not lint */
#include <sys/cdefs.h>
__FBSDID("$FreeBSD: src/lib/libc/stdio/flags.c,v 1.10 2007/01/09 00:28:06 imp Exp $");

#include <sys/types.h>
#include <sys/file.h>
#include <stdio.h>
#include <errno.h>

#include "local.h"

/*
 * Return the (stdio) flags for a given mode.  Store the flags
 * to be passed to an _open() syscall through *optr.
 * Return 0 on error.
 */
int
__sflags(const char *mode, int *optr)
{
	int ret, m, o;

	switch (*mode++) {

	case 'r':	/* open for reading */
		ret = __SRD;
		m = O_RDONLY;
		o = 0;
		break;

	case 'w':	/* open for writing */
		ret = __SWR;
		m = O_WRONLY;
		o = O_CREAT | O_TRUNC;
		break;

	case 'a':	/* open for appending */
		ret = __SWR;
		m = O_WRONLY;
		o = O_CREAT | O_APPEND;
		break;

	default:	/* illegal mode */
		errno = EINVAL;
		return (0);
	}

	/* [rwa]\+ or [rwa]b\+ means read and write */
	if (*mode == 'b')
		mode++;
	if (*mode == '+') {
		ret = __SRW;
		m = O_RDWR;
		mode++;
		if (*mode == 'b')
			mode++;
	}
	if (*mode == 'x')
		o |= O_EXCL;
	*optr = m | o;
	return (ret);
}
