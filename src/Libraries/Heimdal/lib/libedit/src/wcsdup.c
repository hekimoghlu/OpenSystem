/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 13, 2025.
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
/*
 * Copyright (C) 2006 Aleksey Cheusov
 *
 * This material is provided "as is", with absolutely no warranty expressed
 * or implied. Any use is at your own risk.
 *
 * Permission to use or copy this software for any purpose is hereby granted
 * without fee. Permission to modify the code and to distribute modified
 * code is also granted without any restrictions.
 */

#ifndef HAVE_WCSDUP

#include "config.h"

#if defined(LIBC_SCCS) && !defined(lint)
__RCSID("$NetBSD: wcsdup.c,v 1.3 2008/05/26 13:17:48 haad Exp $");
#endif /* LIBC_SCCS and not lint */

#include <stdlib.h>
#include <assert.h>
#include <wchar.h>

wchar_t *
wcsdup(const wchar_t *str)
{
	wchar_t *copy;
	size_t len;

	_DIAGASSERT(str != NULL);

	len = wcslen(str) + 1;
	copy = malloc(len * sizeof (wchar_t));

	if (!copy)
		return NULL;

	return wmemcpy(copy, str, len);
}

#endif

