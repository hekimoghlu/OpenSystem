/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 22, 2022.
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
static char sccsid[] = "@(#)mbrune.c	8.1 (Berkeley) 6/27/93";
#endif /* LIBC_SCCS and not lint */
#include <sys/cdefs.h>
__FBSDID("$FreeBSD: src/lib/libc/locale/mbrune.c,v 1.3 2002/09/18 06:11:21 tjr Exp $");

#include <limits.h>
#include <rune.h>
#include <stddef.h>
#include <string.h>
#include "runedepreciated.h"

char *
mbrune(const char *string, rune_t c)
{
	char const *result;
	rune_t r;
	static int warn_depreciated = 1;

	if (warn_depreciated) {
		warn_depreciated = 0;
		fprintf(stderr, __rune_depreciated_msg, "mbrune");
	}

	while ((r = __sgetrune(string, MB_LEN_MAX, &result))) {
		if (r == c)
			return ((char *)string);
		string = result == string ? string + 1 : result;
	}

	return (c == *string ? (char *)string : NULL);
}

char *
mbrrune(const char *string, rune_t c)
{
	const char *last = 0;
	char const *result;
	rune_t  r;
	static int warn_depreciated = 1;

	if (warn_depreciated) {
		warn_depreciated = 0;
		fprintf(stderr, __rune_depreciated_msg, "mbrrune");
	}

	while ((r = __sgetrune(string, MB_LEN_MAX, &result))) {
		if (r == c)
			last = string;
		string = result == string ? string + 1 : result;
	}
	return (c == *string ? (char *)string : (char *)last);
}

char *
mbmb(const char *string, char *pattern)
{
	rune_t first, r;
	size_t plen, slen;
	char const *result;
	static int warn_depreciated = 1;

	if (warn_depreciated) {
		warn_depreciated = 0;
		fprintf(stderr, __rune_depreciated_msg, "mbmb");
	}

	plen = strlen(pattern);
	slen = strlen(string);
	if (plen > slen)
		return (0);

	first = __sgetrune(pattern, plen, &result);
	if (result == string)
		return (0);

	while (slen >= plen && (r = __sgetrune(string, slen, &result))) {
		if (r == first) {
			if (strncmp(string, pattern, slen) == 0)
				return ((char *) string);
		}
		if (result == string) {
			--slen;
			++string;
		} else {
			slen -= result - string;
			string = result;
		}
	}
	return (0);
}
