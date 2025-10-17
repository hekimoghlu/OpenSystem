/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 15, 2024.
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
static char sccsid[] = "@(#)strcasecmp.c	8.1 (Berkeley) 6/4/93";
#endif /* LIBC_SCCS and not lint */
#include <sys/cdefs.h>
__FBSDID("$FreeBSD: src/lib/libc/string/strcasecmp.c,v 1.8 2009/02/03 17:58:20 danger Exp $");

#include "xlocale_private.h"

#include <strings.h>
#include <ctype.h>

typedef unsigned char u_char;

int
strcasecmp_l(const char *s1, const char *s2, locale_t loc)
{
	const u_char
			*us1 = (const u_char *)s1,
			*us2 = (const u_char *)s2;

	NORMALIZE_LOCALE(loc);
	while (tolower_l(*us1, loc) == tolower_l(*us2++, loc))
		if (*us1++ == '\0')
			return (0);
	return (tolower_l(*us1, loc) - tolower_l(*--us2, loc));
}

int
strcasecmp(const char *s1, const char *s2)
{
	return strcasecmp_l(s1, s2, __current_locale());
}

int
strncasecmp_l(const char *s1, const char *s2, size_t n, locale_t loc)
{
	NORMALIZE_LOCALE(loc);
	if (n != 0) {
		const u_char
				*us1 = (const u_char *)s1,
				*us2 = (const u_char *)s2;

		do {
			if (tolower_l(*us1, loc) != tolower_l(*us2++, loc))
				return (tolower_l(*us1, loc) - tolower_l(*--us2, loc));
			if (*us1++ == '\0')
				break;
		} while (--n != 0);
	}
	return (0);
}

int
strncasecmp(const char *s1, const char *s2, size_t n)
{
	return strncasecmp_l(s1, s2, n, __current_locale());
}
