/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 23, 2022.
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
__FBSDID("$FreeBSD: src/lib/libc/stdtime/timelocal.c,v 1.25 2003/06/13 00:14:07 jkh Exp $");

#include "xlocale_private.h"

#include <stddef.h>
#include <string.h>

#include "ldpart.h"
#include "timelocal.h"

struct xlocale_time {
	struct xlocale_component header;
	char *buffer;
	struct lc_time_T locale;
};

#define LCTIME_SIZE (sizeof(struct lc_time_T) / sizeof(char *))

static const struct lc_time_T	_C_time_locale = {
	{
		"Jan", "Feb", "Mar", "Apr", "May", "Jun",
		"Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
	}, {
		"January", "February", "March", "April", "May", "June",
		"July", "August", "September", "October", "November", "December"
	}, {
		"Sun", "Mon", "Tue", "Wed",
		"Thu", "Fri", "Sat"
	}, {
		"Sunday", "Monday", "Tuesday", "Wednesday",
		"Thursday", "Friday", "Saturday"
	},

	/* X_fmt */
	"%H:%M:%S",

	/*
	 * x_fmt
	 * Since the C language standard calls for
	 * "date, using locale's date format," anything goes.
	 * Using just numbers (as here) makes Quakers happier;
	 * it's also compatible with SVR4.
	 */
	"%m/%d/%y",

	/*
	 * c_fmt
	 */
	"%a %b %e %H:%M:%S %Y",

	/* am */
	"AM",

	/* pm */
	"PM",

	/* date_fmt */
	"%a %b %e %H:%M:%S %Z %Y",
	
	/* alt_month
	 * Standalone months forms for %OB
	 */
	{
		"January", "February", "March", "April", "May", "June",
		"July", "August", "September", "October", "November", "December"
	},

	/* md_order
	 * Month / day order in dates
	 */
	"md",

	/* ampm_fmt
	 * To determine 12-hour clock format time (empty, if N/A)
	 */
	"%I:%M:%S %p"
};

__private_extern__ struct lc_time_T *
__get_current_time_locale(locale_t loc)
{
	return (loc->_time_using_locale
		? &XLOCALE_TIME(loc)->locale
		: (struct lc_time_T *)&_C_time_locale);
}

__private_extern__ int
__time_load_locale(const char *name, locale_t loc)
{
	int ret;
	struct xlocale_time *xp;
	static struct xlocale_time *cache = NULL;

	/* 'name' must be already checked. */
	if (strcmp(name, "C") == 0 || strcmp(name, "POSIX") == 0 ||
	    strncmp(name, "C.", 2) == 0) {
		loc->_time_using_locale = 0;
		xlocale_release(loc->components[XLC_TIME]);
		loc->components[XLC_TIME] = NULL;
		return (_LDP_CACHE);
	}

	/*
	 * If the locale name is the same as our cache, use the cache.
	 */
	if (cache && cache->buffer && strcmp(name, cache->buffer) == 0) {
		loc->_time_using_locale = 1;
		xlocale_release(loc->components[XLC_TIME]);
		loc->components[XLC_TIME] = (void *)cache;
		xlocale_retain(cache);
		return (_LDP_CACHE);
	}
	if ((xp = (struct xlocale_time *)malloc(sizeof(*xp))) == NULL)
		return _LDP_ERROR;
	xp->header.header.retain_count = 1;
	xp->header.header.destructor = destruct_ldpart;
	xp->buffer = NULL;

	ret = __part_load_locale(name, &loc->_time_using_locale,
			&xp->buffer, "LC_TIME",
			LCTIME_SIZE, LCTIME_SIZE,
			(const char **)&xp->locale);
	if (ret == _LDP_LOADED) {
		xlocale_release(loc->components[XLC_TIME]);
		loc->components[XLC_TIME] = (void *)xp;
		xlocale_release(cache);
		cache = xp;
		xlocale_retain(cache);
	} else if (ret == _LDP_ERROR)
		free(xp);

	return (ret);
}
