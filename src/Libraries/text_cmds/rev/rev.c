/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 2, 2025.
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
#ifndef lint
static const char copyright[] =
"@(#) Copyright (c) 1987, 1992, 1993\n\
	The Regents of the University of California.  All rights reserved.\n";
#endif /* not lint */

#if 0
#ifndef lint
static char sccsid[] = "@(#)rev.c	8.3 (Berkeley) 5/4/95";
#endif /* not lint */
#endif

#include <sys/cdefs.h>
__FBSDID("$FreeBSD$");

#include <sys/types.h>

#include <err.h>
#include <errno.h>
#include <locale.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <wchar.h>

static void usage(void);

int
main(int argc, char *argv[])
{
	const char *filename;
	wchar_t *p, *t;
	FILE *fp;
	size_t len;
	int ch, rval;

	setlocale(LC_ALL, "");

	while ((ch = getopt(argc, argv, "")) != -1)
		switch(ch) {
		case '?':
		default:
			usage();
		}

	argc -= optind;
	argv += optind;

	fp = stdin;
	filename = "stdin";
	rval = 0;
	do {
		if (*argv) {
			if ((fp = fopen(*argv, "r")) == NULL) {
				warn("%s", *argv);
				rval = 1;
				++argv;
				continue;
			}
			filename = *argv++;
		}
		while ((p = fgetwln(fp, &len)) != NULL) {
			if (p[len - 1] == '\n')
				--len;
			for (t = p + len - 1; t >= p; --t)
				putwchar(*t);
			putwchar('\n');
		}
		if (ferror(fp)) {
			warn("%s", filename);
			clearerr(fp);
			rval = 1;
		}
		(void)fclose(fp);
	} while(*argv);
	exit(rval);
}

void
usage(void)
{
	(void)fprintf(stderr, "usage: rev [file ...]\n");
	exit(1);
}
