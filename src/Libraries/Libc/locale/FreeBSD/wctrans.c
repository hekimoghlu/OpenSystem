/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 26, 2024.
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
__FBSDID("$FreeBSD: src/lib/libc/locale/wctrans.c,v 1.3 2003/11/01 08:20:58 tjr Exp $");

#include "xlocale_private.h"

#include <errno.h>
#include <string.h>
#include <wctype.h>

enum {
	_WCT_ERROR	= 0,
	_WCT_TOLOWER	= 1,
	_WCT_TOUPPER	= 2
};

wint_t
towctrans_l(wint_t wc, wctrans_t desc, locale_t loc)
{

	NORMALIZE_LOCALE(loc);
	switch (desc) {
	case _WCT_TOLOWER:
		wc = towlower_l(wc, loc);
		break;
	case _WCT_TOUPPER:
		wc = towupper_l(wc, loc);
		break;
	case _WCT_ERROR:
	default:
		errno = EINVAL;
		break;
	}

	return (wc);
}

wint_t
towctrans(wint_t wc, wctrans_t desc)
{
	return towctrans_l(wc, desc, __current_locale());
}

wctrans_t
wctrans(const char *charclass)
{
	struct {
		const char	*name;
		wctrans_t	 trans;
	} ccls[] = {
		{ "tolower",	_WCT_TOLOWER },
		{ "toupper",	_WCT_TOUPPER },
		{ NULL,		_WCT_ERROR },		/* Default */
	};
	int i;

	i = 0;
	while (ccls[i].name != NULL && strcmp(ccls[i].name, charclass) != 0)
		i++;

	if (ccls[i].trans == _WCT_ERROR)
		errno = EINVAL;
	return (ccls[i].trans);
}

/*
 * The extended locale version just calls the regular version.  If there
 * is ever support for arbitrary per-locale translations, this need to
 * be modified.
 */
wctrans_t
wctrans_l(const char *charclass, locale_t loc)
{
	return wctrans(charclass);
}
