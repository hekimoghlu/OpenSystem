/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 24, 2023.
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
"@(#) Copyright (c) 1990, 1993\n\
	The Regents of the University of California.  All rights reserved.\n");
#endif /* not lint */

#ifndef lint
#if 0
static char sccsid[] = "@(#)mkfifo.c	8.2 (Berkeley) 1/5/94";
#endif
#endif /* not lint */
__FBSDID("$FreeBSD$");

#include <sys/types.h>
#include <sys/stat.h>

#include <err.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#ifdef __APPLE__
#include <get_compat.h>
#else
#define COMPAT_MODE(a,b) (1)
#endif /* __APPLE__ */

#define	BASEMODE	(S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP | \
			S_IROTH | S_IWOTH)

static void usage(void);

static int f_mode;

int
main(int argc, char *argv[])
{
	const char *modestr = NULL;
	const void *modep;
	mode_t fifomode;
	int ch, exitval;
#ifdef __APPLE__
	int unix2003compat;
#endif

	while ((ch = getopt(argc, argv, "m:")) != -1)
		switch(ch) {
		case 'm':
			f_mode = 1;
			modestr = optarg;
			break;
		case '?':
		default:
			usage();
		}
	argc -= optind;
	argv += optind;
	if (argv[0] == NULL)
		usage();

#ifdef __APPLE__
	unix2003compat = COMPAT_MODE("bin/mkfifo", "Unix2003");

	fifomode = 0;
	/* The default mode is the value of the bitwise inclusive or of
	   S_IRUSR, S_IWUSR, S_IRGRP, S_IWGRP, S_IROTH, and S_IWOTH
	   modified by the file creation mask */
	if (!unix2003compat)
		fifomode = BASEMODE & ~umask(0);
#endif

	if (f_mode) {
#ifndef __APPLE__
		umask(0);
#endif
		errno = 0;
		if ((modep = setmode(modestr)) == NULL) {
			if (errno)
				err(1, "setmode");
			errx(1, "invalid file mode: %s", modestr);
		}
		fifomode = getmode(modep, BASEMODE);
#ifndef __APPLE__
	} else {
		fifomode = BASEMODE;
#endif
	}

#ifdef __APPLE__
	if (unix2003compat) {
		mode_t maskbits = umask(0);	/* now must disable umask so -m mode won't be masked again */
		if (!f_mode)
			fifomode = BASEMODE & ~maskbits;
	}
#endif

	for (exitval = 0; *argv != NULL; ++argv) {
		if (mkfifo(*argv, fifomode) < 0) {
			warn("%s", *argv);
			exitval = 1;
		}
	}
	exit(exitval);
}

static void
usage(void)
{
	(void)fprintf(stderr, "usage: mkfifo [-m mode] fifo_name ...\n");
	exit(1);
}
