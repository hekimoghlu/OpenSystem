/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 9, 2024.
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

__FBSDID("$FreeBSD$");

#ifndef lint
static const char copyright[] =
"@(#) Copyright (c) 1980, 1988, 1993\n\
	The Regents of the University of California.  All rights reserved.\n";
#endif

#ifndef lint
static const char sccsid[] = "@(#)what.c	8.1 (Berkeley) 6/6/93";
#endif

#include <err.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

static void usage(void);
static bool search(bool, bool, FILE *);

int
main(int argc, char *argv[])
{
	const char *file;
	FILE *in;
	bool found, qflag, sflag;
	int c;

	qflag = sflag = false;

	while ((c = getopt(argc, argv, "qs")) != -1) {
		switch (c) {
		case 'q':
			qflag = true;
			break;
		case 's':
			sflag = true;
			break;
		default:
			usage();
		}
	}
	argc -= optind;
	argv += optind;

	found = false;

	if (argc == 0) {
		if (search(sflag, qflag, stdin))
			found = true;
	} else {
		while (argc--) {
			file = *argv++;
			in = fopen(file, "r");
			if (in == NULL) {
				if (!qflag)
					warn("%s", file);
				continue;
			}
			if (!qflag)
				printf("%s:\n", file);
			if (search(sflag, qflag, in))
				found = true;
			fclose(in);
		}
	}
	exit(found ? 0 : 1);
}

static void
usage(void)
{
	fprintf(stderr, "usage: what [-qs] [file ...]\n");
	exit(1);
}

bool
search(bool one, bool quiet, FILE *in)
{
	bool found;
	int c;

	found = false;

	while ((c = getc(in)) != EOF) {
loop:		if (c != '@')
			continue;
		if ((c = getc(in)) != '(')
			goto loop;
		if ((c = getc(in)) != '#')
			goto loop;
		if ((c = getc(in)) != ')')
			goto loop;
		if (!quiet)
			putchar('\t');
		while ((c = getc(in)) != EOF && c && c != '"' &&
		    c != '>' && c != '\\' && c != '\n')
			putchar(c);
		putchar('\n');
		found = true;
		if (one)
			break;
	}
	return (found);
}
