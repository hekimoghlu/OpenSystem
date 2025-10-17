/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 20, 2023.
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
static const char copyright[] =
"@(#) Copyright (c) 1990, 1993, 1994\n\
	The Regents of the University of California.  All rights reserved.\n";

#if 0
static char sccsid[] = "@(#)main.c	8.4 (Berkeley) 5/4/95";
#endif

#include <sys/cdefs.h>
__FBSDID("$FreeBSD$");

#include <sys/types.h>
#include <sys/stat.h>

#include <err.h>
#include <errno.h>
#include <fcntl.h>
#include <fts.h>
#include <locale.h>
#include <regex.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

#ifdef __APPLE__
#include <get_compat.h>
#endif

#include "find.h"

time_t now;			/* time find was run */
int dotfd;			/* starting directory */
int ftsoptions;			/* options for the ftsopen(3) call */
int ignore_readdir_race;	/* ignore readdir race */
int isdepth;			/* do directories on post-order visit */
int isoutput;			/* user specified output operator */
int issort;         		/* do hierarchies in lexicographical order */
int isxargs;			/* don't permit xargs delimiting chars */
int mindepth = -1, maxdepth = -1; /* minimum and maximum depth */
int regexp_flags = REG_BASIC;	/* use the "basic" regexp by default*/
int exitstatus;

#ifdef __APPLE__
bool unix2003_compat;
#endif

static void usage(void);

int
main(int argc, char *argv[])
{
	char **p, **start;
	int Hflag, Lflag, ch;

	(void)setlocale(LC_ALL, "");

	(void)time(&now);	/* initialize the time-of-day */

#ifdef __APPLE__
	unix2003_compat = COMPAT_MODE("bin/find", "unix2003");
#endif

	p = start = argv;
	Hflag = Lflag = 0;
	ftsoptions = FTS_NOSTAT | FTS_PHYSICAL;
	while ((ch = getopt(argc, argv, "EHLPXdf:sx")) != -1)
		switch (ch) {
		case 'E':
			regexp_flags |= REG_EXTENDED;
			break;
		case 'H':
			Hflag = 1;
			Lflag = 0;
			break;
		case 'L':
			Lflag = 1;
			Hflag = 0;
			break;
		case 'P':
			Hflag = Lflag = 0;
			break;
		case 'X':
			isxargs = 1;
			break;
		case 'd':
			isdepth = 1;
			break;
		case 'f':
			*p++ = optarg;
			break;
		case 's':
			issort = 1;
			break;
		case 'x':
			ftsoptions |= FTS_XDEV;
			break;
		case '?':
		default:
			usage();
		}

	argc -= optind;
	argv += optind;

	if (Hflag)
		ftsoptions |= FTS_COMFOLLOW;
	if (Lflag) {
		ftsoptions &= ~FTS_PHYSICAL;
		ftsoptions |= FTS_LOGICAL;
	}

	/*
	 * Find first option to delimit the file list.  The first argument
	 * that starts with a -, or is a ! or a ( must be interpreted as a
	 * part of the find expression, according to POSIX .2.
	 */
	for (; *argv != NULL; *p++ = *argv++) {
		if (argv[0][0] == '-')
			break;
		if ((argv[0][0] == '!' || argv[0][0] == '(') &&
		    argv[0][1] == '\0')
			break;
	}

	if (p == start)
		usage();
	*p = NULL;

	if ((dotfd = open(".", O_RDONLY | O_CLOEXEC, 0)) < 0)
		ftsoptions |= FTS_NOCHDIR;

#ifdef __APPLE__
	ch = find_execute(find_formplan(argv), start);
	if (ferror(stdout) != 0 || fflush(stdout) != 0)
		err(1, "stdout");
	return (ch);
#else
	exit(find_execute(find_formplan(argv), start));
#endif
}

static void
usage(void)
{
	(void)fprintf(stderr, "%s\n%s\n",
"usage: find [-H | -L | -P] [-EXdsx] [-f path] path ... [expression]",
"       find [-H | -L | -P] [-EXdsx] -f path [path ...] [expression]");
	exit(1);
}
