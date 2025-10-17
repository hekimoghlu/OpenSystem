/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 2, 2021.
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
__FBSDID("$FreeBSD: src/lib/libc/string/strcoll.c,v 1.14 2009/02/03 17:58:20 danger Exp $");

#include "xlocale_private.h"

#include <stdlib.h>
#include <string.h>
#include <wchar.h>
#include <errno.h>
#include "collate.h"

int
strcoll_l(const char *s, const char *s2, locale_t loc)
{
	int ret;
	const wchar_t *t = NULL, *t2 = NULL;
	int sverrno;

	NORMALIZE_LOCALE(loc);
	if (XLOCALE_COLLATE(loc)->__collate_load_error ||
	    (t = __collate_mbstowcs(s, loc)) == NULL ||
	    (t2 = __collate_mbstowcs(s2, loc)) == NULL) {
		sverrno = errno;
		free((void *)t);
		free((void *)t2);
		errno = sverrno;
		return strcmp(s, s2);
	}

	ret = wcscoll_l(t, t2, loc);
	sverrno = errno;
	free((void *)t);
	free((void *)t2);
	errno = sverrno;

	return ret;
}

int
strcoll(const char *s, const char *s2)
{
	return strcoll_l(s, s2, __current_locale());
}
