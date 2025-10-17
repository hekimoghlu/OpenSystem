/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 15, 2022.
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
"@(#) Copyright (c) 1987, 1993\n\
	The Regents of the University of California.  All rights reserved.\n";
#endif /* not lint */

#if 0
#ifndef lint
static char sccsid[] = "@(#)printenv.c	8.2 (Berkeley) 5/4/95";
#endif /* not lint */
#endif

#include <sys/cdefs.h>
__FBSDID("$FreeBSD$");

#include <sys/types.h>

#ifndef __APPLE__
#include <capsicum_helpers.h>
#endif
#include <err.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

void	usage(void);
extern char **environ;

/*
 * printenv
 *
 * Bill Joy, UCB
 * February, 1979
 */
int
main(int argc, char *argv[])
{
	char *cp, **ep;
	size_t len;
	int ch;

#ifndef __APPLE__
	if (caph_limit_stdio() < 0 || caph_enter() < 0)
		err(1, "capsicum");
#endif

	while ((ch = getopt(argc, argv, "")) != -1)
		switch(ch) {
		case '?':
		default:
			usage();
		}
	argc -= optind;
	argv += optind;

	if (argc == 0) {
		for (ep = environ; *ep; ep++)
			(void)printf("%s\n", *ep);
		exit(0);
	}
	len = strlen(*argv);
	for (ep = environ; *ep; ep++)
		if (!memcmp(*ep, *argv, len)) {
			cp = *ep + len;
			if (*cp == '=') {
				(void)printf("%s\n", cp + 1);
				exit(0);
			}
		}
	exit(1);
}

void
usage(void)
{
	(void)fprintf(stderr, "usage: printenv [name]\n");
	exit(1);
}
