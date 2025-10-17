/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 26, 2022.
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
#if 0
static char sccsid[] = "@(#)misc.c	8.2 (Berkeley) 4/1/94";
#endif

#include <sys/cdefs.h>
__FBSDID("$FreeBSD$");

#include <sys/types.h>
#include <sys/stat.h>

#include <err.h>
#include <errno.h>
#include <fts.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "find.h"

/*
 * brace_subst --
 *	Replace occurrences of {} in s1 with s2 and return the result string.
 */
void
brace_subst(char *orig, char **store, char *path, size_t len)
{
	const char *pastorigend, *p, *q;
	char *dst;
	size_t newlen, plen;

	plen = strlen(path);
	newlen = strlen(orig) + 1;
	pastorigend = orig + newlen;
	for (p = orig; (q = strstr(p, "{}")) != NULL; p = q + 2) {
		if (plen > 2 && newlen + plen - 2 < newlen)
			errx(2, "brace_subst overflow");
		newlen += plen - 2;
	}
	if (newlen > len) {
		*store = reallocf(*store, newlen);
		if (*store == NULL)
			err(2, NULL);
	}
	dst = *store;
	for (p = orig; (q = strstr(p, "{}")) != NULL; p = q + 2) {
		memcpy(dst, p, q - p);
		dst += q - p;
		memcpy(dst, path, plen);
		dst += plen;
	}
	memcpy(dst, p, pastorigend - p);
}

/*
 * queryuser --
 *	print a message to standard error and then read input from standard
 *	input. If the input is an affirmative response (according to the
 *	current locale) then 1 is returned.
 */
int
queryuser(char *argv[])
{
	char *p, resp[256];

	(void)fprintf(stderr, "\"%s", *argv);
	while (*++argv)
		(void)fprintf(stderr, " %s", *argv);
	(void)fprintf(stderr, "\"? ");
	(void)fflush(stderr);

	if (fgets(resp, sizeof(resp), stdin) == NULL)
		*resp = '\0';
	if ((p = strchr(resp, '\n')) != NULL)
		*p = '\0';
	else {
		(void)fprintf(stderr, "\n");
		(void)fflush(stderr);
	}
        return (rpmatch(resp) == 1);
}
