/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 27, 2025.
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
#ifndef lint
__COPYRIGHT(
"@(#) Copyright (c) 1989, 1993, 1994\n\
	The Regents of the University of California.  All rights reserved.\n");
#endif /* not lint */

#if 0
#ifndef lint
static char sccsid[] = "@(#)nice.c	8.2 (Berkeley) 4/16/94";
#endif /* not lint */
#endif

__FBSDID("$FreeBSD$");

#include <sys/types.h>
#include <sys/time.h>
#include <sys/resource.h>

#include <ctype.h>
#include <err.h>
#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#ifdef __APPLE__
#include <locale.h>
#endif

#define	DEFNICE	10

void usage(void);

int
main(int argc, char *argv[])
{
	long niceness = DEFNICE;
	int ch;
	char *ep;

#ifdef __APPLE__
	setlocale(LC_ALL, "");
#endif

	/* Obsolescent syntax: -number, --number */
	if (argc >= 2 && argv[1][0] == '-' && (argv[1][1] == '-' ||
	    isdigit((unsigned char)argv[1][1])) && strcmp(argv[1], "--") != 0)
		if (asprintf(&argv[1], "-n%s", argv[1] + 1) < 0)
			err(1, "asprintf");

	while ((ch = getopt(argc, argv, "n:")) != -1) {
		switch (ch) {
		case 'n':
			errno = 0;
			niceness = strtol(optarg, &ep, 10);
			if (ep == optarg || *ep != '\0' || errno ||
			    niceness < INT_MIN || niceness > INT_MAX)
				errx(1, "%s: invalid nice value", optarg);
			break;
		default:
			usage();
		}
	}
	argc -= optind;
	argv += optind;

	if (argc == 0)
		usage();

	errno = 0;
	niceness += getpriority(PRIO_PROCESS, 0);
	if (errno)
		warn("getpriority");
	else if (setpriority(PRIO_PROCESS, 0, (int)niceness))
		warn("setpriority");
	execvp(*argv, argv);
	err(errno == ENOENT ? 127 : 126, "%s", *argv);
}

void
usage(void)
{

	(void)fprintf(stderr,
	    "usage: nice [-n increment] utility [argument ...]\n");
	exit(1);
}
