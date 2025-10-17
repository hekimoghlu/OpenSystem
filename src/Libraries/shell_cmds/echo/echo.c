/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 3, 2024.
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
"@(#) Copyright (c) 1989, 1993\n\
	The Regents of the University of California.  All rights reserved.\n";
#endif /* not lint */

#ifndef lint
static char sccsid[] = "@(#)echo.c	8.1 (Berkeley) 5/31/93";
#endif /* not lint */
#endif
#include <sys/cdefs.h>
__FBSDID("$FreeBSD: src/bin/echo/echo.c,v 1.18 2005/01/10 08:39:22 imp Exp $");

#include <err.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <wchar.h>

static void
flush_and_exit(void)
{
	if (fflush(stdout) != 0)
		err(1, "fflush");
	exit(0);
}

static char *
print_one_char(char *cur, int posix, int *bytes_len_out)
{
	char *next;
	wchar_t wc;
	int bytes_len = mbtowc(&wc, cur, MB_CUR_MAX);
	if (bytes_len <= 0) {
		putchar(*cur);
		bytes_len = 1;
		goto out;
	}

	/* If this is not an escape sequence, just print the character */
	if (wc != '\\') {
		putwchar(wc);
		goto out;
	}

	next = cur + bytes_len;

	if (!posix) {
		/* In non-POSIX mode, the only valid escape sequence is \c */
		if (*next == 'c') {
			flush_and_exit();
		} else {
			putchar(wc);
			goto out;
		}
	} else {
		cur = next;
		bytes_len = 1;
	}

	switch (*cur) {
		case 'a':
			putchar('\a');
			goto out;

		case 'b':
			putchar('\b');
			goto out;

		case 'c':
			flush_and_exit();

		case 'f':
			putchar('\f');
			goto out;

		case 'n':
			putchar('\n');
			goto out;

		case 'r':
			putchar('\r');
			goto out;

		case 't':
			putchar('\t');
			goto out;

		case 'v':
			putchar('\v');
			goto out;

		case '\\':
			putchar('\\');
			goto out;

		case '0': {
			int j = 0, num = 0;
			while ((*++cur >= '0' && *cur <= '7') &&
			       j++ < 3) {
				num <<= 3;
				num |= (*cur - '0');
			}
			putchar(num);
			--cur;
			goto out;
		}
		default:
			--cur;
			putchar(*cur);
			goto out;
	}

 out:
	if (bytes_len_out)
		*bytes_len_out = bytes_len;
	return cur;
}

int
main(int argc, char *argv[])
{
	int nflag = 0;
	int posix = (getenv("POSIXLY_CORRECT") != NULL || getenv("POSIX_PEDANTIC") != NULL);

	if (!posix && argv[1] && strcmp(argv[1], "-n") == 0)
		nflag = 1;

	for (int i = 0; i < argc; i++) {
		/* argv[0] == progname */
		int ignore_arg = (i == 0 || (i == 1 && nflag == 1));
		int last_arg = (i == (argc - 1));
		if (!ignore_arg) {
			char *cur = argv[i];
			size_t arg_len = strlen(cur);
			int bytes_len = 0;

			for (const char *end = cur + arg_len; cur < end; cur += bytes_len) {
				cur = print_one_char(cur, posix, &bytes_len);
			}
		}
		if (last_arg && !nflag)
			putchar('\n');
		else if (!last_arg && !ignore_arg)
			putchar(' ');

		if (fflush(stdout) != 0)
			err(1, "fflush");
	}

	return 0;
}
