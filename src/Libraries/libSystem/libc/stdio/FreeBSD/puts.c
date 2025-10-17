/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 22, 2022.
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
static char sccsid[] = "@(#)puts.c	8.1 (Berkeley) 6/4/93";
#endif /* LIBC_SCCS and not lint */
#include <sys/cdefs.h>
__FBSDID("$FreeBSD: src/lib/libc/stdio/puts.c,v 1.11 2007/01/09 00:28:07 imp Exp $");

#include "namespace.h"
#include <stdio.h>
#include <string.h>
#include "un-namespace.h"
#include "fvwrite.h"
#include "libc_private.h"
#include "local.h"

// 3340719: __puts_null__ is used if string is NULL.  Shared by fputs.c
__private_extern__ char const __puts_null__[] = "(null)";

/*
 * Write the given string to stdout, appending a newline.
 */
int
puts(s)
	char const *s;
{
	int retval;
	size_t c;
	struct __suio uio;
	struct __siov iov[2];

	// 3340719: __puts_null__ is used if s is NULL
	if(s == NULL)
		s = __puts_null__;
	iov[0].iov_base = (void *)s;
	iov[0].iov_len = c = strlen(s);
	iov[1].iov_base = "\n";
	iov[1].iov_len = 1;
	uio.uio_resid = c + 1;
	uio.uio_iov = &iov[0];
	uio.uio_iovcnt = 2;
	FLOCKFILE(stdout);
	ORIENT(stdout, -1);
	retval = __sfvwrite(stdout, &uio) ? EOF : '\n';
	FUNLOCKFILE(stdout);
	return (retval);
}
