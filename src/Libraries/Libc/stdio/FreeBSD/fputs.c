/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 19, 2023.
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
static char sccsid[] = "@(#)fputs.c	8.1 (Berkeley) 6/4/93";
#endif /* LIBC_SCCS and not lint */
#include <sys/cdefs.h>
__FBSDID("$FreeBSD: src/lib/libc/stdio/fputs.c,v 1.12 2007/01/09 00:28:06 imp Exp $");

#include "namespace.h"
#include <stdio.h>
#include <string.h>
#include "un-namespace.h"
#include "fvwrite.h"
#include "libc_private.h"
#include "local.h"
#include "libc_hooks_impl.h"

// 3340719: __puts_null__ is used if string is NULL.  Defined in puts.c
extern char const __puts_null__[];

/*
 * Write the given string to the given file.
 */
int
fputs(const char * __restrict s, FILE * __restrict fp)
{
	int retval;
	struct __suio uio;
	struct __siov iov;

	libc_hooks_will_read_cstring(s);
	libc_hooks_will_write(fp, sizeof(*fp));

	// 3340719: __puts_null__ is used if s is NULL
	if(s == NULL)
		s = __puts_null__;
	iov.iov_base = (void *)s;
	iov.iov_len = uio.uio_resid = strlen(s);
	uio.uio_iov = &iov;
	uio.uio_iovcnt = 1;
	FLOCKFILE(fp);
	ORIENT(fp, -1);
	retval = __sfvwrite(fp, &uio);
	FUNLOCKFILE(fp);
#if __DARWIN_UNIX03
	if (retval == 0)
		return iov.iov_len;
#endif /* __DARWIN_UNIX03 */
	return (retval);
}
