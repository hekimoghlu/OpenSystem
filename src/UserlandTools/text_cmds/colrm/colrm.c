/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 9, 2021.
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
"@(#) Copyright (c) 1991, 1993\n\
	The Regents of the University of California.  All rights reserved.\n";
#endif

#if 0
#ifndef lint
static char sccsid[] = "@(#)colrm.c	8.2 (Berkeley) 5/4/95";
#endif
#endif

#include <sys/cdefs.h>
__FBSDID("$FreeBSD$");

#include <sys/types.h>
#include <err.h>
#include <errno.h>
#include <limits.h>
#include <locale.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <wchar.h>

#define	TAB	8

void check(FILE *);
static void usage(void);

int
main(int argc, char *argv[])
{
	u_long column, start, stop;
	int ch, width;
	char *p;

	setlocale(LC_ALL, "");

	while ((ch = getopt(argc, argv, "")) != -1)
		switch(ch) {
		case '?':
		default:
			usage();
		}
	argc -= optind;
	argv += optind;

	start = stop = 0;
	switch(argc) {
	case 2:
#ifdef __APPLE__
                if(argv[1]) stop = strtol(argv[1], &p, 10);
#else
		stop = strtol(argv[1], &p, 10);
#endif
		if (stop <= 0 || *p)
			errx(1, "illegal column -- %s", argv[1]);
		/* FALLTHROUGH */
	case 1:
#ifdef __APPLE__
	        if(argv[0]) start = strtol(argv[0], &p, 10);
#else
		start = strtol(argv[0], &p, 10);
#endif
		if (start <= 0 || *p)
			errx(1, "illegal column -- %s", argv[0]);
		break;
	case 0:
		break;
	default:
		usage();
	}

	if (stop && start > stop)
		errx(1, "illegal start and stop columns");

	for (column = 0;;) {
		switch (ch = getwchar()) {
		case WEOF:
			check(stdin);
			break;
		case '\b':
			if (column)
				--column;
			break;
		case '\n':
			column = 0;
			break;
		case '\t':
			column = (column + TAB) & ~(TAB - 1);
			break;
		default:
			if ((width = wcwidth(ch)) > 0)
				column += width;
			break;
		}

		if ((!start || column < start || (stop && column > stop)) &&
		    putwchar(ch) == WEOF)
			check(stdout);
	}
}

void
check(FILE *stream)
{
	if (feof(stream))
		exit(0);
	if (ferror(stream))
		err(1, "%s", stream == stdin ? "stdin" : "stdout");
}

void
usage(void)
{
	(void)fprintf(stderr, "usage: colrm [start [stop]]\n");
	exit(1);
}

