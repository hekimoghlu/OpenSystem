/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 13, 2024.
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
"@(#) Copyright (c) 1989, 1991, 1993, 1994\n\
	The Regents of the University of California.  All rights reserved.\n";
#endif /* not lint */

#ifndef lint
static char sccsid[] = "@(#)stty.c	8.3 (Berkeley) 4/2/94";
#endif /* not lint */
#endif
#include <sys/cdefs.h>
__FBSDID("$FreeBSD$");

#include <sys/types.h>

#include <ctype.h>
#include <err.h>
#include <fcntl.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "stty.h"
#include "extern.h"

int
main(int argc, char *argv[])
{
	struct info i;
	enum FMT fmt;
	int ch;
	const char *file, *errstr = NULL;

	fmt = NOTSET;
	i.fd = STDIN_FILENO;
	file = "stdin";

	opterr = 0;
	while (optind < argc &&
	    strspn(argv[optind], "-aefg") == strlen(argv[optind]) &&
	    (ch = getopt(argc, argv, "aef:g")) != -1)
		switch(ch) {
		case 'a':		/* undocumented: POSIX compatibility */
			fmt = POSIX;
			break;
		case 'e':
			fmt = BSD;
			break;
		case 'f':
			if ((i.fd = open(optarg, O_RDONLY | O_NONBLOCK)) < 0)
				err(1, "%s", optarg);
			file = optarg;
			break;
		case 'g':
			fmt = GFLAG;
			break;
		case '?':
		default:
			goto args;
		}

args:	argc -= optind;
	argv += optind;

	if (tcgetattr(i.fd, &i.t) < 0)
		errx(1, "%s isn't a terminal", file);
	if (ioctl(i.fd, TIOCGETD, &i.ldisc) < 0)
		err(1, "TIOCGETD");
	if (ioctl(i.fd, TIOCGWINSZ, &i.win) < 0)
		warn("TIOCGWINSZ");

	checkredirect();			/* conversion aid */

	switch(fmt) {
	case NOTSET:
		if (*argv)
			break;
		/* FALLTHROUGH */
	case BSD:
	case POSIX:
		print(&i.t, &i.win, i.ldisc, fmt);
		break;
	case GFLAG:
		gprint(&i.t, &i.win, i.ldisc);
		break;
	}

	for (i.set = i.wset = 0; *argv; ++argv) {
		if (ksearch(&argv, &i))
			continue;

		if (csearch(&argv, &i))
			continue;

		if (msearch(&argv, &i))
			continue;

		if (isdigit(**argv)) {
			speed_t speed;

			speed = strtonum(*argv, 0, UINT_MAX, &errstr);
			if (errstr)
				err(1, "speed");
			cfsetospeed(&i.t, speed);
			cfsetispeed(&i.t, speed);
			i.set = 1;
			continue;
		}

		if (!strncmp(*argv, "gfmt1", sizeof("gfmt1") - 1)) {
			gread(&i.t, *argv + sizeof("gfmt1") - 1);
			i.set = 1;
			continue;
		}

		warnx("illegal option -- %s", *argv);
		usage();
	}

	if (i.set && tcsetattr(i.fd, 0, &i.t) < 0)
		err(1, "tcsetattr");
	if (i.wset && ioctl(i.fd, TIOCSWINSZ, &i.win) < 0)
		warn("TIOCSWINSZ");
	exit(0);
}

void
usage(void)
{

	(void)fprintf(stderr,
	    "usage: stty [-a | -e | -g] [-f file] [arguments]\n");
	exit (1);
}
