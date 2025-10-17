/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 8, 2023.
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
__FBSDID("$FreeBSD: src/lib/libc/locale/lmessages.c,v 1.14 2003/06/26 10:46:16 phantom Exp $");

#include "xlocale_private.h"

#include <stddef.h>
#include <string.h>

#include "ldpart.h"
#include "lmessages.h"

#define LCMESSAGES_SIZE_FULL (sizeof(struct lc_messages_T) / sizeof(char *))
#define LCMESSAGES_SIZE_MIN \
		(offsetof(struct lc_messages_T, yesstr) / sizeof(char *))

struct xlocale_messages {
	struct xlocale_component header;
	char *buffer;
	struct lc_messages_T locale;
};

static char empty[] = "";

static const struct lc_messages_T _C_messages_locale = {
	"^[yY]" ,	/* yesexpr */
	"^[nN]" ,	/* noexpr */
	"yes" , 	/* yesstr */
	"no"		/* nostr */
};

__private_extern__ int
__messages_load_locale(const char *name, locale_t loc)
{
	int ret;
	struct xlocale_messages *xp;
	static struct xlocale_messages *cache = NULL;

	/* 'name' must be already checked. */
	if (strcmp(name, "C") == 0 || strcmp(name, "POSIX") == 0 ||
	    strncmp(name, "C.", 2) == 0) {
		loc->_messages_using_locale = 0;
		xlocale_release(loc->components[XLC_MESSAGES]);
		loc->components[XLC_MESSAGES] = NULL;
		return (_LDP_CACHE);
	}

	/*
	 * If the locale name is the same as our cache, use the cache.
	 */
	if (cache && cache->buffer && strcmp(name, cache->buffer) == 0) {
		loc->_messages_using_locale = 1;
		xlocale_release(loc->components[XLC_MESSAGES]) ;
		loc->components[XLC_MESSAGES]= (void *)cache;
		xlocale_retain(cache);
		return (_LDP_CACHE);
	}
	if ((xp = (struct xlocale_messages *)malloc(sizeof(*xp))) == NULL)
		return _LDP_ERROR;
	xp->header.header.retain_count = 1;
	xp->header.header.destructor = destruct_ldpart;
	xp->buffer = NULL;

	ret = __part_load_locale(name, &loc->_messages_using_locale,
		  &xp->buffer, "LC_MESSAGES/LC_MESSAGES",
		  LCMESSAGES_SIZE_FULL, LCMESSAGES_SIZE_MIN,
		  (const char **)&xp->locale);
	if (ret == _LDP_LOADED) {
		if (xp->locale.yesstr == NULL)
			xp->locale.yesstr = empty;
		if (xp->locale.nostr == NULL)
			xp->locale.nostr = empty;
		xlocale_release(loc->components[XLC_MESSAGES]);
		loc->components[XLC_MESSAGES] = (void *)xp;
		xlocale_release(cache);
		cache = xp;
		xlocale_retain(cache);
	} else if (ret == _LDP_ERROR)
		free(xp);
	return (ret);
}

__private_extern__ struct lc_messages_T *
__get_current_messages_locale(locale_t loc)
{
	return (loc->_messages_using_locale
		? &XLOCALE_MESSAGES(loc)->locale
		: (struct lc_messages_T *)&_C_messages_locale);
}

#ifdef LOCALE_DEBUG
void
msgdebug() {
locale_t loc = __current_locale();
printf(	"yesexpr = %s\n"
	"noexpr = %s\n"
	"yesstr = %s\n"
	"nostr = %s\n",
	XLOCALE_MESSAGES(loc)->locale.yesexpr,
	XLOCALE_MESSAGES(loc)->locale.noexpr,
	XLOCALE_MESSAGES(loc)->locale.yesstr,
	XLOCALE_MESSAGES(loc)->locale.nostr
);
}
#endif /* LOCALE_DEBUG */
