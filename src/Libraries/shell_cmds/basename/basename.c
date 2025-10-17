/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 25, 2022.
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
"@(#) Copyright (c) 1991, 1993, 1994\n\
	The Regents of the University of California.  All rights reserved.\n";
#endif

#if 0
#ifndef lint
static char sccsid[] = "@(#)basename.c	8.4 (Berkeley) 5/4/95";
#endif /* not lint */
#endif

#include <sys/cdefs.h>
__FBSDID("$FreeBSD$");

#ifndef __APPLE__
#include <capsicum_helpers.h>
#endif
#include <err.h>
#include <libgen.h>
#include <limits.h>
#include <locale.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <wchar.h>

void stripsuffix(char *, const char *, size_t);
void usage(void);

int
main(int argc, char **argv)
{
	char *p, *suffix;
	size_t suffixlen;
	int aflag, ch;

	setlocale(LC_ALL, "");

#ifndef __APPLE__
	if (caph_limit_stdio() < 0 || caph_enter() < 0)
		err(1, "capsicum");
#endif

	aflag = 0;
	suffix = NULL;
	suffixlen = 0;

	while ((ch = getopt(argc, argv, "as:")) != -1)
		switch(ch) {
		case 'a':
			aflag = 1;
			break;
		case 's':
			suffix = optarg;
			break;
		case '?':
		default:
			usage();
		}
	argc -= optind;
	argv += optind;

	if (argc < 1)
		usage();

	if (!*argv[0]) {
		printf("\n");
#ifdef __APPLE__
		if (ferror(stdout) != 0 || fflush(stdout) != 0)
			err(1, "stdout");
#endif
		exit(0);
	}
	if ((p = basename(argv[0])) == NULL)
		err(1, "%s", argv[0]);
	if ((suffix == NULL && !aflag) && argc == 2) {
		suffix = argv[1];
		argc--;
	}
	if (suffix != NULL)
		suffixlen = strlen(suffix);
	while (argc--) {
		if ((p = basename(*argv)) == NULL)
			err(1, "%s", argv[0]);
		stripsuffix(p, suffix, suffixlen);
		argv++;
		(void)printf("%s\n", p);
	}
#ifdef __APPLE__
	if (ferror(stdout) != 0 || fflush(stdout) != 0)
		err(1, "stdout");
#endif
	exit(0);
}

void
stripsuffix(char *p, const char *suffix, size_t suffixlen)
{
	char *q, *r;
	mbstate_t mbs;
	size_t n;

	if (suffixlen && (q = strchr(p, '\0') - suffixlen) > p &&
	    strcmp(suffix, q) == 0) {
		/* Ensure that the match occurred on a character boundary. */
		memset(&mbs, 0, sizeof(mbs));
		for (r = p; r < q; r += n) {
			n = mbrlen(r, MB_LEN_MAX, &mbs);
			if (n == (size_t)-1 || n == (size_t)-2) {
				memset(&mbs, 0, sizeof(mbs));
				n = 1;
			}
		}
		/* Chop off the suffix. */
		if (q == r)
			*q = '\0';
	}
}

void
usage(void)
{

	(void)fprintf(stderr,
"usage: basename string [suffix]\n"
"       basename [-a] [-s suffix] string [...]\n");
	exit(1);
}
