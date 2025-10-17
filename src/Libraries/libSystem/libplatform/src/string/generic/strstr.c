/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 20, 2024.
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
static char sccsid[] = "@(#)strstr.c	8.1 (Berkeley) 6/4/93";
#endif /* LIBC_SCCS and not lint */
#include <sys/cdefs.h>
__FBSDID("$FreeBSD: src/lib/libc/string/strstr.c,v 1.6 2009/02/03 17:58:20 danger Exp $");

#include <sys/_types/_null.h>
#include <platform/string.h>

#if !_PLATFORM_OPTIMIZED_STRSTR

/*
 * Find the first occurrence of find in s.
 */
char *
_platform_strstr(const char *s, const char *find)
{
	char c, sc;
	size_t len;

	if ((c = *find++) != '\0') {
		len = _platform_strlen(find);
		do {
			do {
				if ((sc = *s++) == '\0')
					return (NULL);
			} while (sc != c);
		} while (_platform_strncmp(s, find, len) != 0);
		s--;
	}
	return ((char *)s);
}

#if VARIANT_STATIC
char *
strstr(const char *s, const char *find)
{
	return _platform_strstr(s, find);
}
#endif

#endif
