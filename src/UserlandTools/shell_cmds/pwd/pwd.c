/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 25, 2022.
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
"@(#) Copyright (c) 1991, 1993, 1994\n\
	The Regents of the University of California.  All rights reserved.\n";
#endif /* not lint */

#ifndef lint
static char sccsid[] = "@(#)pwd.c	8.3 (Berkeley) 4/1/94";
#endif /* not lint */
#endif
#include <sys/cdefs.h>
__FBSDID("$FreeBSD$");

#include <sys/param.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <err.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

static char *getcwd_logical(void);
void usage(void);

int
main(int argc, char *argv[])
{
	int physical;
	int ch;
	char *p;

	// 4207130
	physical = 0;
	while ((ch = getopt(argc, argv, "LP")) != -1)
		switch (ch) {
		case 'L':
			physical = 0;
			break;
		case 'P':
			physical = 1;
			break;
		case '?':
		default:
			usage();
		}
	argc -= optind;
	argv += optind;

	if (argc != 0)
		usage();

	/*
	 * If we're trying to find the logical current directory and that
	 * fails, behave as if -P was specified.
	 *
	 * 4701537: Only fall back if $PWD was actually wrong. We want to
	 * fail if the path is too long (errno == ENAMETOOLONG).
	 */
	if ((!physical && (p = getcwd_logical()) != NULL) ||
	    ((physical || errno == ENOENT) && (p = getcwd(NULL, 0)) != NULL))
		printf("%s\n", p);
	else
		err(1, ".");

#ifdef __APPLE__
	if (ferror(stdout) != 0 || fflush(stdout) != 0)
		err(1, "stdout");
#endif
	exit(0);
}

void __dead2
usage(void)
{

	(void)fprintf(stderr, "usage: pwd [-L | -P]\n");
  	exit(1);
}

static char *
getcwd_logical(void)
{
	struct stat lg, phy;
	char *pwd;

	/*
	 * Check that $PWD is an absolute logical pathname referring to
	 * the current working directory.
	 */
	if ((pwd = getenv("PWD")) != NULL && *pwd == '/') {
		if (stat(pwd, &lg) == -1 || stat(".", &phy) == -1)
			return (NULL);
		if (lg.st_dev == phy.st_dev && lg.st_ino == phy.st_ino)
			return (pwd);
	}

	errno = ENOENT;
	return (NULL);
}
