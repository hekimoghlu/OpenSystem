/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 2, 2025.
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
#include "db_config.h"

#include "db_int.h"

/*
 * strcasecmp --
 *	Do strcmp(3) in a case-insensitive manner.
 *
 * PUBLIC: #ifndef HAVE_STRCASECMP
 * PUBLIC: int strcasecmp __P((const char *, const char *));
 * PUBLIC: #endif
 */
int
strcasecmp(s1, s2)
	const char *s1, *s2;
{
	u_char s1ch, s2ch;

	for (;;) {
		s1ch = *s1++;
		s2ch = *s2++;
		if (s1ch >= 'A' && s1ch <= 'Z')		/* tolower() */
			s1ch += 32;
		if (s2ch >= 'A' && s2ch <= 'Z')		/* tolower() */
			s2ch += 32;
		if (s1ch != s2ch)
			return (s1ch - s2ch);
		if (s1ch == '\0')
			return (0);
	}
	/* NOTREACHED */
}

/*
 * strncasecmp --
 *	Do strncmp(3) in a case-insensitive manner.
 *
 * PUBLIC: #ifndef HAVE_STRCASECMP
 * PUBLIC: int strncasecmp __P((const char *, const char *, size_t));
 * PUBLIC: #endif
 */
int
strncasecmp(s1, s2, n)
	const char *s1, *s2;
	register size_t n;
{
	u_char s1ch, s2ch;

	for (; n != 0; --n) {
		s1ch = *s1++;
		s2ch = *s2++;
		if (s1ch >= 'A' && s1ch <= 'Z')		/* tolower() */
			s1ch += 32;
		if (s2ch >= 'A' && s2ch <= 'Z')		/* tolower() */
			s2ch += 32;
		if (s1ch != s2ch)
			return (s1ch - s2ch);
		if (s1ch == '\0')
			return (0);
	}
	return (0);
}
