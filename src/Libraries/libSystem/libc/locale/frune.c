/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 11, 2021.
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
#if defined(LIBC_SCCS) && !defined(lint)
static char sccsid[] = "@(#)frune.c	8.1 (Berkeley) 6/4/93";
#endif /* LIBC_SCCS and not lint */
#include <sys/cdefs.h>
__FBSDID("$FreeBSD: src/lib/libc/locale/frune.c,v 1.3 2002/09/18 06:19:12 tjr Exp $");

#include "xlocale_private.h"

#include <limits.h>
#include <rune.h>
#include <stddef.h>
#include <stdio.h>
#include "runedepreciated.h"

long
fgetrune(fp)
	FILE *fp;
{
	rune_t  r;
	int c, len;
	char buf[MB_LEN_MAX];
	char const *result;
	__darwin_rune_t invalid_rune = __current_locale()->__lc_ctype->_CurrentRuneLocale.__invalid_rune;
	static int warn_depreciated = 1;

	if (warn_depreciated) {
		warn_depreciated = 0;
		fprintf(stderr, __rune_depreciated_msg, "fgetrune");
	}

	len = 0;
	do {
		if ((c = getc(fp)) == EOF) {
			if (len)
				break;
			return (EOF);
		}
		buf[len++] = c;

		if ((r = __sgetrune(buf, len, &result)) != invalid_rune)
			return (r);
	} while (result == buf && len < MB_LEN_MAX);

	while (--len > 0)
		ungetc(buf[len], fp);
	return (invalid_rune);
}

int
fungetrune(r, fp)
	rune_t r;
	FILE* fp;
{
	int len;
	char buf[MB_LEN_MAX];
	static int warn_depreciated = 1;

	if (warn_depreciated) {
		warn_depreciated = 0;
		fprintf(stderr, __rune_depreciated_msg, "fungetrune");
	}

	len = __sputrune(r, buf, MB_LEN_MAX, 0);
	while (len-- > 0)
		if (ungetc(buf[len], fp) == EOF)
			return (EOF);
	return (0);
}

int
fputrune(r, fp)
	rune_t r;
	FILE *fp;
{
	int i, len;
	char buf[MB_LEN_MAX];
	static int warn_depreciated = 1;

	if (warn_depreciated) {
		warn_depreciated = 0;
		fprintf(stderr, __rune_depreciated_msg, "fputrune");
	}

	len = __sputrune(r, buf, MB_LEN_MAX, 0);

	for (i = 0; i < len; ++i)
		if (putc(buf[i], fp) == EOF)
			return (EOF);

	return (0);
}
