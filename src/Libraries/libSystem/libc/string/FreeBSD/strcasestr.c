/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 21, 2024.
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
__FBSDID("$FreeBSD: src/lib/libc/string/strcasestr.c,v 1.5 2009/02/03 17:58:20 danger Exp $");

#include "xlocale_private.h"

#include <ctype.h>
#include <string.h>

/*
 * Find the first occurrence of find in s, ignore case.
 */
char *
strcasestr_l(s, find, loc)
	const char *s, *find;
	locale_t loc;
{
	char c, sc;
	size_t len;

	NORMALIZE_LOCALE(loc);
	if ((c = *find++) != 0) {
		c = tolower_l((unsigned char)c, loc);
		len = strlen(find);
		do {
			do {
				if ((sc = *s++) == 0)
					return (NULL);
			} while ((char)tolower_l((unsigned char)sc, loc) != c);
		} while (strncasecmp_l(s, find, len, loc) != 0);
		s--;
	}
	return ((char *)s);
}

char *
strcasestr(const char *s, const char *find)
{
	return strcasestr_l(s, find, __current_locale());
}
