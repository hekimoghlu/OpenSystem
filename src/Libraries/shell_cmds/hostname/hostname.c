/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 28, 2022.
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
"@(#) Copyright (c) 1988, 1993\n\
	The Regents of the University of California.  All rights reserved.\n";
#endif /* not lint */

#ifndef lint
static char sccsid[] = "@(#)hostname.c	8.1 (Berkeley) 5/31/93";
#endif /* not lint */
#endif
#include <sys/cdefs.h>
__FBSDID("$FreeBSD$");

#include <sys/param.h>

#include <err.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static void usage(void) __dead2;

int
main(int argc, char *argv[])
{
	int ch, sflag, dflag;
	char hostname[MAXHOSTNAMELEN], *hostp, *p;

	sflag = 0;
	dflag = 0;
	while ((ch = getopt(argc, argv, "fsd")) != -1)
		switch (ch) {
		case 'f':
			/*
			 * On Linux, "hostname -f" prints FQDN.
			 * BSD "hostname" always prints FQDN by
			 * default, so we accept but ignore -f.
			 */
			break;
		case 's':
			sflag = 1;
			break;
		case 'd':
			dflag = 1;
			break;
		case '?':
		default:
			usage();
		}
	argc -= optind;
	argv += optind;

	if (argc > 1 || (sflag && dflag))
		usage();

	if (*argv) {
		if (sethostname(*argv, (int)strlen(*argv)))
			err(1, "sethostname");
	} else {
		hostp = hostname;
		if (gethostname(hostname, (int)sizeof(hostname)))
			err(1, "gethostname");
		if (sflag) {
			p = strchr(hostname, '.');
			if (p != NULL)
				*p = '\0';
		} else if (dflag) {
			p = strchr(hostname, '.');
			if (p != NULL)
				hostp = p + 1;
		}
		(void)printf("%s\n", hostp);
	}
	exit(0);
}

static void
usage(void)
{

	(void)fprintf(stderr, "usage: hostname [-f] [-s | -d] [name-of-host]\n");
	exit(1);
}
