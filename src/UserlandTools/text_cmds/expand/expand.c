/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 12, 2023.
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
"@(#) Copyright (c) 1980, 1993\n\
	The Regents of the University of California.  All rights reserved.\n";
#endif /* not lint */

#ifndef lint
#if 0
static char sccsid[] = "@(#)expand.c	8.1 (Berkeley) 6/9/93";
#endif
#endif /* not lint */
#include <sys/cdefs.h>
__FBSDID("$FreeBSD$");

#include <ctype.h>
#include <err.h>
#include <locale.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <wchar.h>
#include <wctype.h>

/*
 * expand - expand tabs to equivalent spaces
 */
static int	nstops;
static int	tabstops[100];

static void getstops(char *);
static void usage(void) __dead2; 

int
main(int argc, char *argv[])
{
	const char *curfile;
	wint_t wc;
	int c, column;
	int n;
	int rval;
	int width;

	setlocale(LC_CTYPE, "");

	/* handle obsolete syntax */
	while (argc > 1 && argv[1] && argv[1][0] == '-' &&
	    isdigit((unsigned char)argv[1][1])) {
		getstops(&argv[1][1]);
		argc--; argv++;
	}

	while ((c = getopt (argc, argv, "t:")) != -1) {
		switch (c) {
		case 't':
			getstops(optarg);
			break;
		case '?':
		default:
			usage();
			/* NOTREACHED */
		}
	}
	argc -= optind;
	argv += optind;

	rval = 0;
	do {
		if (argc > 0 && *argv) {
			if (freopen(argv[0], "r", stdin) == NULL) {
				warn("%s", argv[0]);
				rval = 1;
				break;
			}
			curfile = argv[0];
			argc--, argv++;
		} else
			curfile = "stdin";
		column = 0;
		while ((wc = getwchar()) != WEOF) {
			switch (wc) {
			case '\t':
				if (nstops == 0) {
					do {
						putwchar(' ');
						column++;
					} while (column & 07);
					continue;
				}
				if (nstops == 1) {
					do {
						putwchar(' ');
						column++;
					} while (((column - 1) % tabstops[0]) != (tabstops[0] - 1));
					continue;
				}
				for (n = 0; n < nstops; n++)
					if (tabstops[n] > column)
						break;
				if (n == nstops) {
					putwchar(' ');
					column++;
					continue;
				}
				while (column < tabstops[n]) {
					putwchar(' ');
					column++;
				}
				continue;

			case '\b':
				if (column)
					column--;
				putwchar('\b');
				continue;

			default:
				putwchar(wc);
				if ((width = wcwidth(wc)) > 0)
					column += width;
				continue;

			case '\n':
				putwchar(wc);
				column = 0;
				continue;
			}
		}
		if (ferror(stdin)) {
			warn("%s", curfile);
			rval = 1;
		}
	} while (argc > 0);
#ifdef __APPLE__
	if (ferror(stdout) != 0 || fflush(stdout) != 0)
		err(1, "stdout");
#endif
	exit(rval);
}

static void
getstops(char *cp)
{
	int i;

	nstops = 0;
	for (;;) {
		i = 0;
		while (*cp >= '0' && *cp <= '9')
			i = i * 10 + *cp++ - '0';
		if (i <= 0)
			errx(1, "bad tab stop spec");
		if (nstops > 0 && i <= tabstops[nstops-1])
			errx(1, "bad tab stop spec");
		if (nstops == sizeof(tabstops) / sizeof(*tabstops))
			errx(1, "too many tab stops");
		tabstops[nstops++] = i;
		if (*cp == 0)
			break;
		if (*cp != ',' && !isblank((unsigned char)*cp))
			errx(1, "bad tab stop spec");
		cp++;
	}
}

static void
usage(void)
{
	(void)fprintf (stderr, "usage: expand [-t tablist] [file ...]\n");
	exit(1);
}
