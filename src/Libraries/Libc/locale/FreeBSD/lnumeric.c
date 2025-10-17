/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 20, 2022.
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
__FBSDID("$FreeBSD: src/lib/libc/locale/lnumeric.c,v 1.16 2003/06/26 10:46:16 phantom Exp $");

#include "xlocale_private.h"

#include <limits.h>
#include <string.h>

#include "ldpart.h"
#include "lnumeric.h"

extern const char *__fix_locale_grouping_str(const char *);

#define LCNUMERIC_SIZE (sizeof(struct lc_numeric_T) / sizeof(char *))

static const struct lc_numeric_T _C_numeric_locale = {
	".",     	/* decimal_point */
	"",     	/* thousands_sep */
	""		/* grouping [C99 7.11.2.1]*/
};

__private_extern__ int
__numeric_load_locale(const char *name, locale_t loc)
{
	int ret;
	struct xlocale_numeric *xp;
	static struct xlocale_numeric *cache = NULL;

	/* 'name' must be already checked. */
	if (strcmp(name, "C") == 0 || strcmp(name, "POSIX") == 0 ||
	    strncmp(name, "C.", 2) == 0) {
		if (!loc->_numeric_using_locale)
			return (_LDP_CACHE);
		loc->_numeric_using_locale = 0;
		xlocale_release(loc->components[XLC_NUMERIC]);
		loc->components[XLC_NUMERIC] = NULL;
		loc->__nlocale_changed = 1;
		return (_LDP_CACHE);
	}

	if (loc->_numeric_using_locale && strcmp(name, XLOCALE_NUMERIC(loc)->buffer) == 0)
		return (_LDP_CACHE);
	/*
	 * If the locale name is the same as our cache, use the cache.
	 */
	if (cache && cache->buffer && strcmp(name, cache->buffer) == 0) {
		loc->_numeric_using_locale = 1;
		xlocale_release(loc->components[XLC_NUMERIC]);
		loc->components[XLC_NUMERIC] = (void *)cache;
		xlocale_retain(cache);
		loc->__nlocale_changed = 1;
		return (_LDP_CACHE);
	}
	if ((xp = (struct xlocale_numeric *)malloc(sizeof(*xp))) == NULL)
		return _LDP_ERROR;
	xp->header.header.retain_count = 1;
	xp->header.header.destructor = destruct_ldpart;
	xp->buffer = NULL;

	ret = __part_load_locale(name, &loc->_numeric_using_locale,
		&xp->buffer, "LC_NUMERIC",
		LCNUMERIC_SIZE, LCNUMERIC_SIZE,
		(const char **)&xp->locale);
	if (ret != _LDP_ERROR)
		loc->__nlocale_changed = 1;
	else
		free(xp);
	if (ret == _LDP_LOADED) {
		/* Can't be empty according to C99 */
		if (*xp->locale.decimal_point == '\0')
			xp->locale.decimal_point =
			    _C_numeric_locale.decimal_point;
		xp->locale.grouping =
		    __fix_locale_grouping_str(xp->locale.grouping);
		xlocale_release(loc->components[XLC_NUMERIC]);
		loc->components[XLC_NUMERIC] = (void *)xp;
		xlocale_release(cache);
		cache = xp;
		xlocale_retain(cache);
	}
	return (ret);
}

__private_extern__ struct lc_numeric_T *
__get_current_numeric_locale(locale_t loc)
{
	return (loc->_numeric_using_locale
		? &XLOCALE_NUMERIC(loc)->locale
		: (struct lc_numeric_T *)&_C_numeric_locale);
}

#ifdef LOCALE_DEBUG
void
numericdebug(void) {
locale_t loc = __current_locale();
printf(	"decimal_point = %s\n"
	"thousands_sep = %s\n"
	"grouping = %s\n",
	XLOCALE_NUMERIC(loc)->locale.decimal_point,
	XLOCALE_NUMERIC(loc)->locale.thousands_sep,
	XLOCALE_NUMERIC(loc)->locale.grouping
);
}
#endif /* LOCALE_DEBUG */
