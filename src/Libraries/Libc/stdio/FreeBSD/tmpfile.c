/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 25, 2025.
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
static char sccsid[] = "@(#)tmpfile.c	8.1 (Berkeley) 6/4/93";
#endif /* LIBC_SCCS and not lint */
#include <sys/cdefs.h>
__FBSDID("$FreeBSD: src/lib/libc/stdio/tmpfile.c,v 1.10 2007/01/09 00:28:07 imp Exp $");

#include "namespace.h"
#include <sys/types.h>
#include <signal.h>
#include <unistd.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <paths.h>
#include "un-namespace.h"

FILE *
tmpfile(void)
{
	sigset_t set, oset;
	FILE *fp;
	int fd, sverrno;
#define	TRAILER	"tmp.XXXXXX"
	char *buf;
	const char *tmpdir;

	tmpdir = NULL;
	if (issetugid() == 0)
		tmpdir = getenv("TMPDIR");
	if (tmpdir == NULL)
		tmpdir = _PATH_TMP;

	if (*tmpdir == '\0')
		return (NULL);

	(void)asprintf(&buf, "%s%s%s", tmpdir,
	    (tmpdir[strlen(tmpdir) - 1] == '/') ? "" : "/", TRAILER);
	if (buf == NULL)
		return (NULL);

	sigfillset(&set);
	(void)_sigprocmask(SIG_BLOCK, &set, &oset);

	fd = mkstemp(buf);
	if (fd != -1)
		(void)unlink(buf);

	free(buf);

	(void)_sigprocmask(SIG_SETMASK, &oset, NULL);

	if (fd == -1)
		return (NULL);

	if ((fp = fdopen(fd, "w+")) == NULL) {
		sverrno = errno;
		(void)_close(fd);
		errno = sverrno;
		return (NULL);
	}
	return (fp);
}
