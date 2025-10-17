/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 19, 2023.
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
#include <sys/types.h>

#include <err.h>
#include <limits.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "extern.h"

int
c_link(const char *file1, off_t skip1, const char *file2, off_t skip2,
    off_t limit)
{
	char buf1[PATH_MAX], *p1;
	char buf2[PATH_MAX], *p2;
	ssize_t len1, len2;
	int dfound;
	off_t byte;
	u_char ch;

	if ((len1 = readlink(file1, buf1, sizeof(buf1) - 1)) < 0) {
		if (!sflag)
			err(ERR_EXIT, "%s", file1);
		else
			exit(ERR_EXIT);
	}

	if ((len2 = readlink(file2, buf2, sizeof(buf2) - 1)) < 0) {
		if (!sflag)
			err(ERR_EXIT, "%s", file2);
		else
			exit(ERR_EXIT);
	}

	if (skip1 > len1)
		skip1 = len1;
	buf1[len1] = '\0';

	if (skip2 > len2)
		skip2 = len2;
	buf2[len2] = '\0';

	dfound = 0;
	byte = 1;
	for (p1 = buf1 + skip1, p2 = buf2 + skip2;
	    *p1 && *p2 && (limit == 0 || byte <= limit); p1++, p2++) {
		if ((ch = *p1) != *p2) {
			if (xflag) {
				dfound = 1;
				(void)printf("%08llx %02x %02x\n",
				    (long long)byte - 1, ch, *p2);
			} else if (lflag) {
				dfound = 1;
				if (bflag)
					(void)printf("%6lld %3o %c %3o %c\n",
					    (long long)byte, ch, ch, *p2, *p2);
				else
					(void)printf("%6lld %3o %3o\n",
					    (long long)byte, ch, *p2);
			} else {
				diffmsg(file1, file2, byte, 1, ch, *p2);
				return (DIFF_EXIT);
			}
		}
		byte++;
	}

	if (*p1 || *p2) {
		eofmsg (*p1 ? file2 : file1);
		return (DIFF_EXIT);
	}
	return (dfound ? DIFF_EXIT : 0);
}
