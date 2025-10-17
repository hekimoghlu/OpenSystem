/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 30, 2023.
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
__RCSID("$NetBSD: wcslen.c,v 1.1 2000/12/23 23:14:36 itojun Exp $");
#endif /* LIBC_SCCS and not lint */
#endif
__FBSDID("$FreeBSD$");

#include <wchar.h>

size_t
wcslen(const wchar_t *s)
{
	const wchar_t *p;

	p = s;
	while (*p)
		p++;

	return p - s;
}
