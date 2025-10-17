/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 22, 2023.
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
	struct __xlocale_st_numeric *xp;
	static struct __xlocale_st_numeric *cache = NULL;

	/* 'name' must be already checked. */
	if (strcmp(name, "C") == 0 || strcmp(name, "POSIX") == 0) {
		if (!loc->_numeric_using_locale)
			return (_LDP_CACHE);
		loc->_numeric_using_locale = 0;
		XL_RELEASE(loc->__lc_numeric);
		loc->__lc_numeric = NULL;
		loc->__nlocale_changed = 1;
		return (_LDP_CACHE);
	}

	if (loc->_numeric_using_locale && strcmp(name, loc->__lc_numeric->_numeric_locale_buf) == 0)
		return (_LDP_CACHE);
	/*
	 * If the locale name is the same as our cache, use the cache.
	 */
	if (cache && cache->_numeric_locale_buf && strcmp(name, cache->_numeric_locale_buf) == 0) {
		loc->_numeric_using_locale = 1;
		XL_RELEASE(loc->__lc_numeric);
		loc->__lc_numeric = cache;
		XL_RETAIN(loc->__lc_numeric);
		loc->__nlocale_changed = 1;
		return (_LDP_CACHE);
	}
	if ((xp = (struct __xlocale_st_numeric *)malloc(sizeof(*xp))) == NULL)
		return _LDP_ERROR;
	xp->__refcount = 1;
	xp->__free_extra = (__free_extra_t)__ldpart_free_extra;
	xp->_numeric_locale_buf = NULL;

	ret = __part_load_locale(name, &loc->_numeric_using_locale,
		&xp->_numeric_locale_buf, "LC_NUMERIC",
		LCNUMERIC_SIZE, LCNUMERIC_SIZE,
		(const char **)&xp->_numeric_locale);
	if (ret != _LDP_ERROR)
		loc->__nlocale_changed = 1;
	else
		free(xp);
	if (ret == _LDP_LOADED) {
		/* Can't be empty according to C99 */
		if (*xp->_numeric_locale.decimal_point == '\0')
			xp->_numeric_locale.decimal_point =
			    _C_numeric_locale.decimal_point;
		xp->_numeric_locale.grouping =
		    __fix_locale_grouping_str(xp->_numeric_locale.grouping);
		XL_RELEASE(loc->__lc_numeric);
		loc->__lc_numeric = xp;
		XL_RELEASE(cache);
		cache = xp;
		XL_RETAIN(cache);
	}
	return (ret);
}

__private_extern__ struct lc_numeric_T *
__get_current_numeric_locale(locale_t loc)
{
	return (loc->_numeric_using_locale
		? &loc->__lc_numeric->_numeric_locale
		: (struct lc_numeric_T *)&_C_numeric_locale);
}

#ifdef LOCALE_DEBUG
void
numericdebug(void) {
locale_t loc = __current_locale();
printf(	"decimal_point = %s\n"
	"thousands_sep = %s\n"
	"grouping = %s\n",
	loc->__lc_numeric->_numeric_locale.decimal_point,
	loc->__lc_numeric->_numeric_locale.thousands_sep,
	loc->__lc_numeric->_numeric_locale.grouping
);
}
#endif /* LOCALE_DEBUG */
