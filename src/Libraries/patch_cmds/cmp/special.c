/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 25, 2023.
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

#ifndef __APPLE__
#include <capsicum_helpers.h>
#endif
#include <err.h>
#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>

#include "extern.h"

int
c_special(int fd1, const char *file1, off_t skip1,
    int fd2, const char *file2, off_t skip2, off_t limit)
{
	int ch1, ch2;
	off_t byte, line;
	FILE *fp1, *fp2;
	int dfound;

#ifndef __APPLE__
	if (caph_limit_stream(fd1, CAPH_READ) < 0)
		err(ERR_EXIT, "caph_limit_stream(%s)", file1);
	if (caph_limit_stream(fd2, CAPH_READ) < 0)
		err(ERR_EXIT, "caph_limit_stream(%s)", file2);
	if (caph_enter() < 0)
		err(ERR_EXIT, "unable to enter capability mode");
#endif

	if ((fp1 = fdopen(fd1, "r")) == NULL)
		err(ERR_EXIT, "%s", file1);
	(void)setvbuf(fp1, NULL, _IOFBF, 65536);
	if ((fp2 = fdopen(fd2, "r")) == NULL)
		err(ERR_EXIT, "%s", file2);
	(void)setvbuf(fp2, NULL, _IOFBF, 65536);

	dfound = 0;
	while (skip1--)
		if (getc(fp1) == EOF)
			goto eof;
	while (skip2--)
		if (getc(fp2) == EOF)
			goto eof;

	for (byte = line = 1; limit == 0 || byte <= limit; ++byte) {
#ifdef SIGINFO
		if (info) {
			(void)fprintf(stderr, "%s %s char %zu line %zu\n",
			    file1, file2, (size_t)byte, (size_t)line);
			info = 0;
		}
#endif
		ch1 = getc(fp1);
		ch2 = getc(fp2);
		if (ch1 == EOF || ch2 == EOF)
			break;
		if (ch1 != ch2) {
			if (xflag) {
				dfound = 1;
				(void)printf("%08llx %02x %02x\n",
				    (long long)byte - 1, ch1, ch2);
			} else if (lflag) {
				dfound = 1;
				if (bflag)
					(void)printf("%6lld %3o %c %3o %c\n",
					    (long long)byte, ch1, ch1, ch2,
					    ch2);
				else
					(void)printf("%6lld %3o %3o\n",
					    (long long)byte, ch1, ch2);
			} else {
				diffmsg(file1, file2, byte, line, ch1, ch2);
				return (DIFF_EXIT);
			}
		}
		if (ch1 == '\n')
			++line;
	}

eof:	if (ferror(fp1))
		err(ERR_EXIT, "%s", file1);
	if (ferror(fp2))
		err(ERR_EXIT, "%s", file2);
	if (feof(fp1)) {
		if (!feof(fp2)) {
			eofmsg(file1);
			return (DIFF_EXIT);
		}
	} else {
		if (feof(fp2)) {
			eofmsg(file2);
			return (DIFF_EXIT);
		}
	}
	fclose(fp2);
	fclose(fp1);
	return (dfound ? DIFF_EXIT : 0);
}
