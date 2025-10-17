/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 19, 2022.
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
static char sccsid[] = "@(#)putw.c	8.1 (Berkeley) 6/4/93";
#endif /* LIBC_SCCS and not lint */
#include <sys/cdefs.h>
__FBSDID("$FreeBSD: src/lib/libc/stdio/putw.c,v 1.10 2007/01/09 00:28:07 imp Exp $");

#include "namespace.h"
#include <stdio.h>
#include "un-namespace.h"
#include "fvwrite.h"
#include "libc_private.h"

int
putw(w, fp)
	int w;
	FILE *fp;
{
	int retval;
	struct __suio uio;
	struct __siov iov;

	iov.iov_base = &w;
	iov.iov_len = uio.uio_resid = sizeof(w);
	uio.uio_iov = &iov;
	uio.uio_iovcnt = 1;
	FLOCKFILE(fp);
	retval = __sfvwrite(fp, &uio);
	FUNLOCKFILE(fp);
	return (retval);
}
