/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 25, 2025.
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
#include <sys/cdefs.h> /* __used */
#ifndef lint
__used static const char copyright[] =
"@(#) Copyright (c) 1991, 1993\n\
	The Regents of the University of California.  All rights reserved.\n";
#endif /* not lint */

#ifndef lint
#if 0
static char sccsid[] = "@(#)cksum.c	8.2 (Berkeley) 4/28/95";
#endif
#endif /* not lint */

#include <sys/cdefs.h>
__FBSDID("$FreeBSD$");

#include <sys/types.h>

#include <err.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "extern.h"

static void usage(void);

int
main(int argc, char **argv)
{
	uint32_t val;
	int ch, fd, rval;
	off_t len;
	char *fn, *p;
	int (*cfncn)(int, uint32_t *, off_t *);
	void (*pfncn)(char *, uint32_t, off_t);

	if ((p = strrchr(argv[0], '/')) == NULL)
		p = argv[0];
	else
		++p;
	if (!strcmp(p, "sum")) {
		cfncn = csum1;
		pfncn = psum1;
		++argv;
	} else {
		cfncn = crc;
		pfncn = pcrc;

		while ((ch = getopt(argc, argv, "o:")) != -1)
			switch (ch) {
			case 'o':
				if (!strcmp(optarg, "1")) {
					cfncn = csum1;
					pfncn = psum1;
				} else if (!strcmp(optarg, "2")) {
					cfncn = csum2;
					pfncn = psum2;
				} else if (!strcmp(optarg, "3")) {
					cfncn = crc32;
					pfncn = pcrc;
				} else {
					warnx("illegal argument to -o option");
					usage();
				}
				break;
			case '?':
			default:
				usage();
			}
		argc -= optind;
		argv += optind;
	}

	fd = STDIN_FILENO;
	fn = NULL;
	rval = 0;
	do {
		if (*argv) {
			fn = *argv++;
			if ((fd = open(fn, O_RDONLY, 0)) < 0) {
				warn("%s", fn);
				rval = 1;
				continue;
			}
		}
		if (cfncn(fd, &val, &len)) {
			warn("%s", fn ? fn : "stdin");
			rval = 1;
		} else
			pfncn(fn, val, len);
		(void)close(fd);
	} while (*argv);
#ifdef __APPLE__
	if (rval == 0 && (ferror(stdout) != 0 || fflush(stdout) != 0))
		err(1, "stdout");
#endif
	exit(rval);
}

static void
usage(void)
{
	(void)fprintf(stderr, "usage: cksum [-o 1 | 2 | 3] [file ...]\n");
	(void)fprintf(stderr, "       sum [file ...]\n");
	exit(1);
}
