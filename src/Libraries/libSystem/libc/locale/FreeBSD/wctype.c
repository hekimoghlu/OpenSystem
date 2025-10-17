/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 27, 2023.
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
__FBSDID("$FreeBSD: src/lib/libc/locale/wctype.c,v 1.4 2008/03/17 18:22:23 antoine Exp $");

#include "xlocale_private.h"

#include <ctype.h>
#include <string.h>
#include <wctype.h>
#include <limits.h>

wctype_t
wctype_l(const char *property, locale_t loc)
{
	_RuneLocale *rl;
	static const struct {
		const char	*name;
		wctype_t	 mask;
	} props[] = {
		{ "alnum",	_CTYPE_A|_CTYPE_D },
		{ "alpha",	_CTYPE_A },
		{ "blank",	_CTYPE_B },
		{ "cntrl",	_CTYPE_C },
		{ "digit",	_CTYPE_D },
		{ "graph",	_CTYPE_G },
		{ "lower",	_CTYPE_L },
		{ "print",	_CTYPE_R },
		{ "punct",	_CTYPE_P },
		{ "space",	_CTYPE_S },
		{ "upper",	_CTYPE_U },
		{ "xdigit",	_CTYPE_X },
		{ "ideogram",	_CTYPE_I },	/* BSD extension */
		{ "special",	_CTYPE_T },	/* BSD extension */
		{ "phonogram",	_CTYPE_Q },	/* BSD extension */
		{ "rune",	0xFFFFFF00L },	/* BSD extension */
		{ NULL,		0UL },		/* Default */
	};
	int i;

	i = 0;
	while (props[i].name != NULL && strcmp(props[i].name, property) != 0)
		i++;

	if (props[i].mask)
		return (props[i].mask);

	NORMALIZE_LOCALE(loc);
	rl = &loc->__lc_ctype->_CurrentRuneLocale;
	if ((i = rl->__ncharclasses) > 0) {
		_RuneCharClass *rp;
		for (rp = rl->__charclasses; i-- > 0; rp++) {
			if (strncmp(rp->__name, property, CHARCLASS_NAME_MAX) == 0)
				return (rp->__mask);
		}
	}
	return 0;
}

wctype_t
wctype(const char *property)
{
	return wctype_l(property, __current_locale());
}
