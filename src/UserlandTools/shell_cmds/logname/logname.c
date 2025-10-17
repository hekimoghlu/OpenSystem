/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 15, 2025.
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
"@(#) Copyright (c) 1991, 1993, 1994\n\
	The Regents of the University of California.  All rights reserved.\n";
#endif /* not lint */

#ifndef lint
static const char sccsid[] = "@(#)logname.c	8.2 (Berkeley) 4/3/94";
#endif /* not lint */
#include <sys/cdefs.h>
__FBSDID("$FreeBSD$");

#ifndef __APPLE__
#include <capsicum_helpers.h>
#endif
#include <err.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>

void usage(void);

int
main(int argc, char *argv[] __unused)
{
	char *p;

#ifndef __APPLE__
    if (caph_limit_stdio() < 0 || caph_enter() < 0)
		err(1, "capsicum");
#endif
	if (argc != 1)
		usage();
	if ((p = getlogin()) == NULL)
		err(1, NULL);
	(void)printf("%s\n", p);
#ifdef __APPLE__
	if (ferror(stdout) != 0 || fflush(stdout) != 0)
		err(1, "stdout");
#endif
	exit(0);
}

void
usage(void)
{
	(void)fprintf(stderr, "usage: logname\n");
	exit(1);
}
