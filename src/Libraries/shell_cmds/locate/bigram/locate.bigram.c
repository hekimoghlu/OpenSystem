/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 10, 2025.
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
static char copyright[] =
"@(#) Copyright (c) 1989, 1993\n\
	The Regents of the University of California.  All rights reserved.\n";
#endif /* not lint */

#ifndef lint
static char sccsid[] = "@(#)locate.bigram.c	8.1 (Berkeley) 6/6/93";
#endif /* not lint */
#endif

/*
 *  bigram < sorted_file_names | sort -nr | 
 *  	awk 'NR <= 128 { printf $2 }' > bigrams
 *
 * List bigrams for 'updatedb' script.
 * Use 'code' to encode a file using this output.
 */

#ifndef __APPLE__
#include <capsicum_helpers.h>
#endif
#include <err.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/param.h>			/* for MAXPATHLEN */
#include "locate.h"

u_char buf1[MAXPATHLEN] = " ";
u_char buf2[MAXPATHLEN];
u_int bigram[UCHAR_MAX + 1][UCHAR_MAX + 1];

int
main(void)
{
	u_char *cp;
	u_char *oldpath = buf1, *path = buf2;
	u_int i, j;

#ifndef __APPLE__
	if (caph_limit_stdio() < 0 || caph_enter() < 0)
		err(1, "capsicum");
#endif

     	while (fgets(path, sizeof(buf2), stdin) != NULL) {

		/* 
		 * We don't need remove newline character '\n'.
		 * '\n' is less than ASCII_MIN and will be later
		 * ignored at output.
		 */


		/* skip longest common prefix */
		for (cp = path; *cp == *oldpath; cp++, oldpath++)
			if (*cp == '\0')
				break;

		while (*cp != '\0' && *(cp + 1) != '\0') {
			bigram[(u_char)*cp][(u_char)*(cp + 1)]++;
			cp += 2;
		}

		/* swap pointers */
		if (path == buf1) { 
			path = buf2;
			oldpath = buf1;
		} else {
			path = buf1;
			oldpath = buf2;
		}
   	}

	/* output, boundary check */
	for (i = ASCII_MIN; i <= ASCII_MAX; i++)
		for (j = ASCII_MIN; j <= ASCII_MAX; j++)
			if (bigram[i][j] != 0)
				(void)printf("%4u %c%c\n", bigram[i][j], i, j);

	exit(0);
}
