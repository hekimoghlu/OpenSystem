/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 3, 2022.
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
#if defined(LIBC_SCCS) && !defined(lint)
static char sccsid[] = "@(#)strcmp.c	8.1 (Berkeley) 6/4/93";
#if 0
__RCSID("$NetBSD: wcscmp.c,v 1.3 2001/01/05 12:13:12 itojun Exp $");
#endif
#endif /* LIBC_SCCS and not lint */
__FBSDID("$FreeBSD: src/lib/libc/string/wcscmp.c,v 1.9 2009/02/03 17:58:20 danger Exp $");

#include <wchar.h>

/*
 * Compare strings.
 */
int
wcscmp(const wchar_t *s1, const wchar_t *s2)
{

	while (*s1 == *s2++)
		if (*s1++ == '\0')
			return (0);
	/* XXX assumes wchar_t = int */
	return (*(const unsigned int *)s1 - *(const unsigned int *)--s2);
}
