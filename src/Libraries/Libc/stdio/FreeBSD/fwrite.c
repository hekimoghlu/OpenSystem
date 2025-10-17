/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 22, 2023.
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
static char sccsid[] = "@(#)fwrite.c	8.1 (Berkeley) 6/4/93";
#endif /* LIBC_SCCS and not lint */
#include <sys/cdefs.h>
__FBSDID("$FreeBSD: src/lib/libc/stdio/fwrite.c,v 1.13 2009/07/12 13:09:43 ed Exp $");

#include "namespace.h"
#include <stdio.h>
#include "un-namespace.h"
#include "local.h"
#include "fvwrite.h"
#include "libc_private.h"
#include "libc_hooks_impl.h"

/*
 * The maximum amount to write to avoid integer overflow (especially for
 * uio_resid in struct __suio).  INT_MAX is odd, so it make sense to make it
 * even.  We subtract (BUFSIZ - 1) to get a whole number of BUFSIZ chunks.
 */
#define MAXWRITE	(INT_MAX - (BUFSIZ - 1))

/*
 * Write `count' objects (each size `size') from memory to the given file.
 * Return the number of whole objects written.
 */
size_t
fwrite(const void * __restrict buf, size_t size, size_t count,
    FILE * __restrict fp)
{
	size_t n, resid;
	struct __suio uio;
	struct __siov iov;
	int s;

	/*
	 * ANSI and SUSv2 require a return value of 0 if size or count are 0.
	 */
	n = count * size;
#if __DARWIN_UNIX03
	if (n == 0)
		return (0);
#endif
	uio.uio_iov = &iov;
	uio.uio_iovcnt = 1;

	libc_hooks_will_write(fp, sizeof(*fp));

	FLOCKFILE(fp);
	ORIENT(fp, -1);

	for (resid = n; resid > 0; buf += s, resid -= s) {
		s = resid > INT_MAX ? MAXWRITE : (int)resid;
		iov.iov_base = (void *)buf;
		uio.uio_resid = iov.iov_len = s;

		libc_hooks_will_read(buf, s);

		/*
		 * The usual case is success (__sfvwrite returns 0);
		 * skip the divide if this happens, since divides are
		 * generally slow and since this occurs whenever size==0.
		 */
		if (__sfvwrite(fp, &uio) != 0) {
			count = (n - resid + s - uio.uio_resid) / size;
			break;
		}
	}
	FUNLOCKFILE(fp);
	return (count);
}
