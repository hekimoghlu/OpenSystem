/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 23, 2024.
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
"@(#) Copyright (c) 1988, 1993\n\
	The Regents of the University of California.  All rights reserved.\n";
#endif /* not lint */

#ifndef lint
#if 0
static char sccsid[] = "@(#)tty.c	8.1 (Berkeley) 6/6/93";
#endif
#endif /* not lint */

#include <sys/cdefs.h>
__FBSDID("$FreeBSD$");

#include <err.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

static void usage(void);

int
main(int argc, char *argv[])
{
	int ch, sflag;
	char *t;

	sflag = 0;
	while ((ch = getopt(argc, argv, "s")) != -1)
		switch (ch) {
		case 's':
			sflag = 1;
			break;
		case '?':
		default:
			usage();
			/* NOTREACHED */
		}

	t = ttyname(STDIN_FILENO);
	if (!sflag)
		puts(t ? t : "not a tty");
#ifdef __APPLE__
	if (t && (ferror(stdout) != 0 || fflush(stdout) != 0))
		err(2, "stdout");
#endif
	exit(t ? EXIT_SUCCESS : EXIT_FAILURE);
}

static void
usage(void)
{
	fprintf(stderr, "usage: %s [-s]\n", getprogname());
	exit(2);
}
