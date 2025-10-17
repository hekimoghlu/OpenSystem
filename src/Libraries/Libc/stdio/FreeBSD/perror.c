/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 15, 2023.
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
static char sccsid[] = "@(#)perror.c	8.1 (Berkeley) 6/4/93";
#endif /* LIBC_SCCS and not lint */
#include <sys/cdefs.h>
__FBSDID("$FreeBSD: src/lib/libc/stdio/perror.c,v 1.9 2007/01/09 00:28:07 imp Exp $");

#include "namespace.h"
#include <sys/types.h>
#include <sys/uio.h>
#include <unistd.h>
#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <string.h>
#include "un-namespace.h"
#include "libc_private.h"
#include "local.h"

void
perror(const char *s)
{
	char msgbuf[NL_TEXTMAX];
	struct iovec *v;
	struct iovec iov[4];

	v = iov;
	if (s != NULL && *s != '\0') {
		v->iov_base = (char *)s;
		v->iov_len = strlen(s);
		v++;
		v->iov_base = ": ";
		v->iov_len = 2;
		v++;
	}
	strerror_r(errno, msgbuf, sizeof(msgbuf));
	v->iov_base = msgbuf;
	v->iov_len = strlen(v->iov_base);
	v++;
	v->iov_base = "\n";
	v->iov_len = 1;
	FLOCKFILE(stderr);
	__sflush(stderr);
	(void)_writev(stderr->_file, iov, (v - iov) + 1);
	stderr->_flags &= ~__SOFF;
	FUNLOCKFILE(stderr);
}
