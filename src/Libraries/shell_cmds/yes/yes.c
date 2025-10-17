/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 29, 2023.
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

#ifndef lint
#if 0
static char sccsid[] = "@(#)yes.c	8.1 (Berkeley) 6/6/93";
#else
static const char rcsid[] = "$FreeBSD$";
#endif
#endif /* not lint */

#ifndef __APPLE__
#include <capsicum_helpers.h>
#endif
#include <err.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

int
main(int argc, char **argv)
{
	char buf[8192];
	char y[2] = { 'y', '\n' };
	char * exp = y;
	size_t buflen = 0;
	size_t explen = sizeof(y);
	size_t more;
	ssize_t ret;

#ifndef __APPLE__
	if (caph_limit_stdio() < 0 || caph_enter() < 0)
		err(1, "capsicum");
#endif

	if (argc > 1) {
		exp = argv[1];
		explen = strlen(exp) + 1;
		exp[explen - 1] = '\n';
	}

	if (explen <= sizeof(buf)) {
		while (buflen < sizeof(buf) - explen) {
			memcpy(buf + buflen, exp, explen);
			buflen += explen;
		}
		exp = buf;
		explen = buflen;
	}

	more = explen;
	while ((ret = write(STDOUT_FILENO, exp + (explen - more), more)) > 0)
		if ((more -= ret) == 0)
			more = explen;

	err(1, "stdout");
	/*NOTREACHED*/
}
