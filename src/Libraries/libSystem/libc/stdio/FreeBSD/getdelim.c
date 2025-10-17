/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 22, 2022.
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
#include <sys/cdefs.h>
__FBSDID("$FreeBSD: src/lib/libc/stdio/getdelim.c,v 1.3 2009/10/04 19:43:36 das Exp $");

#include "namespace.h"
#include <os/overflow.h>
#include <sys/param.h>
#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "un-namespace.h"

#include "libc_private.h"
#include "local.h"

static inline size_t
p2roundup(size_t n)
{

	if (!powerof2(n)) {
		n--;
		n |= n >> 1;
		n |= n >> 2;
		n |= n >> 4;
		n |= n >> 8;
		n |= n >> 16;
#if SIZE_T_MAX > 0xffffffffU
		n |= n >> 32;
#endif
		n++;
	}
	return (n);
}

/*
 * Expand *linep to hold len bytes (up to SSIZE_MAX + 1).
 */
static inline int
expandtofit(char ** __restrict linep, size_t len, size_t * __restrict capp)
{
	char *newline;
	size_t newcap;

	if (len > (size_t)SSIZE_MAX + 1) {
		errno = EOVERFLOW;
		return (-1);
	}
	if (len > *capp) {
		if (len == (size_t)SSIZE_MAX + 1)	/* avoid overflow */
			newcap = (size_t)SSIZE_MAX + 1;
		else
			newcap = p2roundup(len);
		newline = realloc(*linep, newcap);
		if (newline == NULL)
			return (-1);
		*capp = newcap;
		*linep = newline;
	}
	return (0);
}

/*
 * Append the src buffer to the *dstp buffer. The buffers are of
 * length srclen and *dstlenp, respectively, and dst has space for
 * *dstlenp bytes. After the call, *dstlenp and *dstcapp are updated
 * appropriately, and *dstp is reallocated if needed. Returns 0 on
 * success, -1 on allocation failure.
 */
static int
sappend(char ** __restrict dstp, size_t * __restrict dstlenp,
	size_t * __restrict dstcapp, char * __restrict src, size_t srclen)
{
	size_t tmp;

	/* avoid overflowing the result length */
	if (os_add3_overflow(srclen, *dstlenp, 1, &tmp)) {
		errno = EOVERFLOW;
		return (-1);
	}

	/* ensure room for srclen + dstlen + terminating NUL */
	if (expandtofit(dstp, tmp, dstcapp))
		return (-1);
	memcpy(*dstp + *dstlenp, src, srclen);
	*dstlenp += srclen;
	return (0);
}

ssize_t
getdelim(char ** __restrict linep, size_t * __restrict linecapp, int delim,
	 FILE * __restrict fp)
{
	u_char *endp;
	size_t linelen;

	FLOCKFILE(fp);
	ORIENT(fp, -1);

	if (linep == NULL || linecapp == NULL) {
		errno = EINVAL;
		goto error;
	}

	if (*linep == NULL)
		*linecapp = 0;

	if (fp->_r <= 0 && __srefill(fp)) {
		/* If fp is at EOF already, we just need space for the NUL. */
		if (__sferror(fp) || expandtofit(linep, 1, linecapp))
			goto error;
		FUNLOCKFILE(fp);
		(*linep)[0] = '\0';
		return (-1);
	}

	linelen = 0;
	while ((endp = memchr(fp->_p, delim, fp->_r)) == NULL) {
		if (sappend(linep, &linelen, linecapp, (char*)fp->_p, fp->_r))
			goto error;
		if (__srefill(fp)) {
			if (__sferror(fp))
				goto error;
			goto done;	/* hit EOF */
		}
	}
	endp++;	/* snarf the delimiter, too */
	if (sappend(linep, &linelen, linecapp, (char*)fp->_p, endp - fp->_p))
		goto error;
	fp->_r -= endp - fp->_p;
	fp->_p = endp;
done:
	/* Invariant: *linep has space for at least linelen+1 bytes. */
	(*linep)[linelen] = '\0';
	FUNLOCKFILE(fp);
	return (linelen);

error:
	fp->_flags |= __SERR;
	FUNLOCKFILE(fp);
	return (-1);
}
