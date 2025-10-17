/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 8, 2022.
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
#if 0
#if defined(LIBC_SCCS) && !defined(lint)
__RCSID("$NetBSD: wmemchr.c,v 1.1 2000/12/23 23:14:37 itojun Exp $");
#endif /* LIBC_SCCS and not lint */
#endif
__FBSDID("$FreeBSD: src/lib/libc/string/wmemchr.c,v 1.7 2009/02/03 17:58:20 danger Exp $");

#include <wchar.h>

wchar_t	*
wmemchr(const wchar_t *s, wchar_t c, size_t n)
{
	size_t i;

	for (i = 0; i < n; i++) {
		if (*s == c) {
			/* LINTED const castaway */
			return (wchar_t *)s;
		}
		s++;
	}
	return NULL;
}
