/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 19, 2023.
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
__FBSDID("$FreeBSD: head/bin/sh/bltin/echo.c 326025 2017-11-20 19:49:47Z pfg $");

/*
 * Echo command.
 */

#define main echocmd

#include "bltin.h"

/* #define eflag 1 */

int
main(int argc, char *argv[])
{
	char **ap;
	char *p;
	char c;
	int count;
	int nflag = 0;
#ifndef eflag
	int eflag = 0;
#endif

	ap = argv;
	if (argc)
		ap++;
	if ((p = *ap) != NULL) {
		if (equal(p, "-n")) {
			nflag++;
			ap++;
		} else if (equal(p, "-e")) {
#ifndef eflag
			eflag++;
#endif
			ap++;
		}
	}
	while ((p = *ap++) != NULL) {
		while ((c = *p++) != '\0') {
			if (c == '\\' && eflag) {
				switch (*p++) {
				case 'a':  c = '\a';  break;
				case 'b':  c = '\b';  break;
				case 'c':  return 0;		/* exit */
				case 'e':  c = '\033';  break;
				case 'f':  c = '\f';  break;
				case 'n':  c = '\n';  break;
				case 'r':  c = '\r';  break;
				case 't':  c = '\t';  break;
				case 'v':  c = '\v';  break;
				case '\\':  break;		/* c = '\\' */
				case '0':
					c = 0;
					count = 3;
					while (--count >= 0 && (unsigned)(*p - '0') < 8)
						c = (c << 3) + (*p++ - '0');
					break;
				default:
					p--;
					break;
				}
			}
			putchar(c);
		}
		if (*ap)
			putchar(' ');
	}
	if (! nflag)
		putchar('\n');
	return 0;
}
