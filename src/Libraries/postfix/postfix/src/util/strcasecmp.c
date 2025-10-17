/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 12, 2024.
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
#endif					/* LIBC_SCCS and not lint */

#include <sys_defs.h>
#include <ctype.h>

int     strcasecmp(const char *s1, const char *s2)
{
    const unsigned char *us1 = (const unsigned char *) s1;
    const unsigned char *us2 = (const unsigned char *) s2;

    while (tolower(*us1) == tolower(*us2++))
	if (*us1++ == '\0')
	    return (0);
    return (tolower(*us1) - tolower(*--us2));
}

int     strncasecmp(const char *s1, const char *s2, size_t n)
{
    if (n != 0) {
	const unsigned char *us1 = (const unsigned char *) s1;
	const unsigned char *us2 = (const unsigned char *) s2;

	do {
	    if (tolower(*us1) != tolower(*us2++))
		return (tolower(*us1) - tolower(*--us2));
	    if (*us1++ == '\0')
		break;
	} while (--n != 0);
    }
    return (0);
}
