/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 25, 2023.
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
static char sccsid[] = "@(#)tempnam.c	8.1 (Berkeley) 6/4/93";
#endif /* LIBC_SCCS and not lint */
#include <sys/cdefs.h>
__FBSDID("$FreeBSD: src/lib/libc/stdio/tempnam.c,v 1.11 2007/01/09 00:28:07 imp Exp $");

#include <sys/param.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <paths.h>

#include "libc_hooks_impl.h"

__warn_references(tempnam,
    "warning: tempnam() possibly used unsafely; consider using mkstemp()");

extern char *_mktemp(char *);

char *
tempnam(const char *dir, const char *pfx)
{
	int sverrno;
	char *f, *name;

	libc_hooks_will_read_cstring(dir);
	libc_hooks_will_read_cstring(pfx);

	if (!(name = malloc(MAXPATHLEN))) {
		return(NULL);
	}

	if (!pfx)
		pfx = "tmp.";

#if !__DARWIN_UNIX03
	if (issetugid() == 0 && (f = getenv("TMPDIR"))) {
		(void)snprintf(name, MAXPATHLEN, "%s%s%sXXXXXX", f,
		    *(f + strlen(f) - 1) == '/'? "": "/", pfx);
		if ((f = _mktemp(name))) {
			return(f);
		}
	}
#endif /* !__DARWIN_UNIX03 */
	if ((f = (char *)dir)) {
#if __DARWIN_UNIX03
	    if (access(dir, W_OK) == 0) {
#endif /* __DARWIN_UNIX03 */
		(void)snprintf(name, MAXPATHLEN, "%s%s%sXXXXXX", f,
		    *(f + strlen(f) - 1) == '/'? "": "/", pfx);
		if ((f = _mktemp(name))) {
			return(f);
		}
#if __DARWIN_UNIX03
	    }
#endif /* __DARWIN_UNIX03 */
	}

	f = P_tmpdir;
#if __DARWIN_UNIX03
	if (access(f, W_OK) == 0) {	/* directory accessible? */
#endif /* __DARWIN_UNIX03 */
	(void)snprintf(name, MAXPATHLEN, "%s%sXXXXXX", f, pfx);
	if ((f = _mktemp(name))) {
		return(f);
	}

#if __DARWIN_UNIX03
	}
	if (issetugid() == 0 && (f = getenv("TMPDIR")) && access(f, W_OK) == 0) {
		(void)snprintf(name, MAXPATHLEN, "%s%s%sXXXXXX", f,
		    *(f + strlen(f) - 1) == '/'? "": "/", pfx);
		if ((f = _mktemp(name))) {
			return(f);
		}
	}
#endif /* __DARWIN_UNIX03 */
	f = _PATH_TMP;
	(void)snprintf(name, MAXPATHLEN, "%s%sXXXXXX", f, pfx);
	if ((f = _mktemp(name))) {
		return(f);
	}

	sverrno = errno;
	free(name);
	errno = sverrno;
	return(NULL);
}
