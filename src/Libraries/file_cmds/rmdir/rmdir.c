/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 26, 2023.
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
#ifndef lint
static char const copyright[] =
"@(#) Copyright (c) 1992, 1993, 1994\n\
	The Regents of the University of California.  All rights reserved.\n";
#endif /* not lint */

#ifndef lint
static char sccsid[] = "@(#)rmdir.c	8.3 (Berkeley) 4/2/94";
#endif /* not lint */
#endif
#include <sys/cdefs.h>
__FBSDID("$FreeBSD$");

#include <err.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int rm_path(char *);
static void usage(void);

static int pflag;
static int vflag;

int
main(int argc, char *argv[])
{
	int ch, errors;

	while ((ch = getopt(argc, argv, "pv")) != -1)
		switch(ch) {
		case 'p':
			pflag = 1;
			break;
		case 'v':
			vflag = 1;
			break;
		case '?':
		default:
			usage();
		}
	argc -= optind;
	argv += optind;

	if (argc == 0)
		usage();

	for (errors = 0; *argv; argv++) {
		if (rmdir(*argv) < 0) {
			warn("%s", *argv);
			errors = 1;
		} else {
			if (vflag)
				printf("%s\n", *argv);
			if (pflag)
				errors |= rm_path(*argv);
		}
	}

	exit(errors);
}

static int
rm_path(char *path)
{
	char *p;

	p = path + strlen(path);
	while (--p > path && *p == '/')
		;
	*++p = '\0';
	while ((p = strrchr(path, '/')) != NULL) {
		/* Delete trailing slashes. */
		while (--p >= path && *p == '/')
			;
		*++p = '\0';
		if (p == path)
			break;

		if (rmdir(path) < 0) {
			warn("%s", path);
			return (1);
		}
		if (vflag)
			printf("%s\n", path);
	}

	return (0);
}

static void
usage(void)
{

	(void)fprintf(stderr, "usage: rmdir [-pv] directory ...\n");
	exit(1);
}
