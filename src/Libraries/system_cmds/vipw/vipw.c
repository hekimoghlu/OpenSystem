/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 4, 2024.
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
static const char copyright[] =
"@(#) Copyright (c) 1987, 1993, 1994\n\
	The Regents of the University of California.  All rights reserved.\n";
#endif /* not lint */

#ifndef lint
static char sccsid[] = "@(#)vipw.c	8.3 (Berkeley) 4/2/94";
#endif /* not lint */
#endif
#include <sys/cdefs.h>
__FBSDID("$FreeBSD$");

#include <sys/types.h>
#include <sys/stat.h>

#include <err.h>
#include <pwd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#ifdef __APPLE__
#include "pw_util.h"
#else
#include <libutil.h>		/* must be after pwd.h */
#endif

static void	usage(void);

int
main(int argc, char *argv[])
{
	const char *passwd_dir = NULL;
	int ch, pfd, tfd;
	char *line;
	size_t len;

	while ((ch = getopt(argc, argv, "d:")) != -1)
		switch (ch) {
		case 'd':
			passwd_dir = optarg;
			break;
		case '?':
		default:
			usage();
		}

	argc -= optind;
	argv += optind;

	if (argc != 0)
		usage();

	if (pw_init(passwd_dir, NULL) == -1)
		err(1, "pw_init()");
	if ((pfd = pw_lock()) == -1) {
		pw_fini();
		err(1, "pw_lock()");
	}
	if ((tfd = pw_tmp(pfd)) == -1) {
		pw_fini();
		err(1, "pw_tmp()");
	}
	(void)close(tfd);
	/* Force umask for partial writes made in the edit phase */
	(void)umask(077);

	for (;;) {
		switch (pw_edit(0)) {
		case -1:
			pw_fini();
			err(1, "pw_edit()");
		case 0:
			pw_fini();
			errx(0, "no changes made");
		default:
			break;
		}
		if (pw_mkdb(NULL) == 0) {
			pw_fini();
			errx(0, "password list updated");
		}
		printf("re-edit the password file? ");
		fflush(stdout);
		if ((line = fgetln(stdin, &len)) == NULL) {
			pw_fini();
			err(1, "fgetln()");
		}
		if (len > 0 && (*line == 'N' || *line == 'n'))
			break;
	}
	pw_fini();
	exit(0);
}

static void
usage(void)
{

	(void)fprintf(stderr, "usage: vipw [-d directory]\n");
	exit(1);
}
