/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 21, 2025.
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
/*
 * Copyright (c) 1997, 2004 Todd C. Miller <Todd.Miller@courtesan.com>
 *
 * Permission to use, copy, modify, and distribute this software for any
 * purpose with or without fee is hereby granted, provided that the above
 * copyright notice and this permission notice appear in all copies.
 *
 * THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
 * WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
 * ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
 * WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
 * ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
 * OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
 */

#include <sys/cdefs.h>
__FBSDID("$FreeBSD: src/lib/libc/gen/dirname.c,v 1.8 2008/11/03 05:19:45 delphij Exp $");

#include <errno.h>
#include <libgen.h>
#include <stdlib.h>
#include <string.h>
#include <sys/param.h>

char *
dirname_r(const char *path, char *dname)
{
	const char *endp;
	size_t len;

	/* Empty or NULL string gets treated as "." */
	if (path == NULL || *path == '\0') {
		dname[0] = '.';
		dname[1] = '\0';
		return (dname);
	}

	/* Strip any trailing slashes */
	endp = path + strlen(path) - 1;
	while (endp > path && *endp == '/')
		endp--;

	/* Find the start of the dir */
	while (endp > path && *endp != '/')
		endp--;

	/* Either the dir is "/" or there are no slashes */
	if (endp == path) {
		dname[0] = *endp == '/' ? '/' : '.';
		dname[1] = '\0';
		return (dname);
	} else {
		/* Move forward past the separating slashes */
		do {
			endp--;
		} while (endp > path && *endp == '/');
	}

	len = endp - path + 1;
	if (len >= MAXPATHLEN) {
		errno = ENAMETOOLONG;
		return (NULL);
	}
	memmove(dname, path, len);
	dname[len] = '\0';
	return (dname);
}

#if __DARWIN_UNIX03
#define const /**/
#endif

char *
dirname(const char *path)
{
	static char *dname = NULL;

	if (dname == NULL) {
		dname = (char *)malloc(MAXPATHLEN);
		if (dname == NULL)
			return (NULL);
	}
	return (dirname_r(path, dname));
}

