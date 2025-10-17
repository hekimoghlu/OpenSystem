/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 30, 2024.
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
__FBSDID("$FreeBSD: src/lib/libc/locale/setrunelocale.c,v 1.51 2008/01/23 03:05:35 ache Exp $");

#include "xlocale_private.h"

#include <runetype.h>
#include <errno.h>
#include <limits.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <wchar.h>
#include "ldpart.h"
#include "mblocal.h"
#include "setlocale.h"

extern struct __xlocale_st_runelocale	*_Read_RuneMagi(FILE *);

#ifdef UNIFDEF_LEGACY_RUNE_APIS
/* depreciated interfaces */
rune_t	sgetrune(const char *, size_t, char const **);
int	sputrune(rune_t, char *, size_t, char **);
#endif /* UNIFDEF_LEGACY_RUNE_APIS */

__private_extern__ int
__setrunelocale(const char *encoding, locale_t loc)
{
	FILE *fp;
	char name[PATH_MAX];
	struct __xlocale_st_runelocale *xrl;
	_RuneLocale *rl;
	int saverr, ret;
	static struct __xlocale_st_runelocale *CachedRuneLocale;
	extern int __mb_cur_max;
	extern int __mb_sb_limit;
	static os_unfair_lock cache_lock = OS_UNFAIR_LOCK_INIT;

	/*
	 * The "C" and "POSIX" locale are always here.
	 */
	if (strcmp(encoding, "C") == 0 || strcmp(encoding, "POSIX") == 0) {
		XL_RELEASE(loc->__lc_ctype);
		loc->__lc_ctype = &_DefaultRuneXLocale;
		/* no need to retain _DefaultRuneXLocale */
		if (loc == &__global_locale) {
			_CurrentRuneLocale = &loc->__lc_ctype->_CurrentRuneLocale;
			__mb_cur_max = loc->__lc_ctype->__mb_cur_max;
			__mb_sb_limit = loc->__lc_ctype->__mb_sb_limit;
		}
		return (0);
	}

	/*
	 * If the locale name is the same as our cache, use the cache.
	 */
	os_unfair_lock_lock(&cache_lock);
	if (CachedRuneLocale != NULL &&
	    strcmp(encoding, CachedRuneLocale->__ctype_encoding) == 0) {
		XL_RELEASE(loc->__lc_ctype);
		loc->__lc_ctype = CachedRuneLocale;
		XL_RETAIN(loc->__lc_ctype);
		if (loc == &__global_locale) {
			_CurrentRuneLocale = &loc->__lc_ctype->_CurrentRuneLocale;
			__mb_cur_max = loc->__lc_ctype->__mb_cur_max;
			__mb_sb_limit = loc->__lc_ctype->__mb_sb_limit;
		}
		os_unfair_lock_unlock(&cache_lock);
		return (0);
	}
	os_unfair_lock_unlock(&cache_lock);

	/*
	 * Slurp the locale file into the cache.
	 */

	/* Range checking not needed, encoding length already checked before */
	(void) strcpy(name, encoding);
	(void) strcat(name, "/LC_CTYPE");

	if ((fp = fdopen(__open_path_locale(name), "r")) == NULL)
		return (errno == 0 ? ENOENT : errno);

	if ((xrl = _Read_RuneMagi(fp)) == NULL) {
		saverr = (errno == 0 ? EFTYPE : errno);
		(void)fclose(fp);
		return (saverr);
	}
	(void)fclose(fp);

	xrl->__mbrtowc = NULL;
	xrl->__mbsinit = NULL;
	xrl->__mbsnrtowcs = __mbsnrtowcs_std;
	xrl->__wcrtomb = NULL;
	xrl->__wcsnrtombs = __wcsnrtombs_std;

	rl = &xrl->_CurrentRuneLocale;

#ifdef UNIFDEF_LEGACY_RUNE_APIS
	/* provide backwards compatibility (depreciated interface) */
	rl->__sputrune = sputrune;
	rl->__sgetrune = sgetrune;
#else /* UNIFDEF_LEGACY_RUNE_APIS */
	rl->__sputrune = NULL;
	rl->__sgetrune = NULL;
#endif /* UNIFDEF_LEGACY_RUNE_APIS */

	if (strcmp(rl->__encoding, "NONE") == 0)
		ret = _none_init(xrl);
	else if (strcmp(rl->__encoding, "ASCII") == 0)
		ret = _ascii_init(xrl);
	else if (strcmp(rl->__encoding, "UTF-8") == 0)
		ret = _UTF8_init(xrl);
	else if (strcmp(rl->__encoding, "EUC") == 0)
		ret = _EUC_init(xrl);
	else if (strcmp(rl->__encoding, "GB18030") == 0)
 		ret = _GB18030_init(xrl);
	else if (strcmp(rl->__encoding, "GB2312") == 0)
		ret = _GB2312_init(xrl);
	else if (strcmp(rl->__encoding, "GBK") == 0)
		ret = _GBK_init(xrl);
	else if (strcmp(rl->__encoding, "BIG5") == 0)
		ret = _BIG5_init(xrl);
	else if (strcmp(rl->__encoding, "MSKanji") == 0)
		ret = _MSKanji_init(xrl);
	else if (strcmp(rl->__encoding, "UTF2") == 0)
		ret = _UTF2_init(xrl);
	else
		ret = EFTYPE;

	if (ret == 0) {
		(void)strcpy(xrl->__ctype_encoding, encoding);
		XL_RELEASE(loc->__lc_ctype);
		loc->__lc_ctype = xrl;
		if (loc == &__global_locale) {
			_CurrentRuneLocale = &loc->__lc_ctype->_CurrentRuneLocale;
			__mb_cur_max = loc->__lc_ctype->__mb_cur_max;
			__mb_sb_limit = loc->__lc_ctype->__mb_sb_limit;
		}
		os_unfair_lock_lock(&cache_lock);
		XL_RELEASE(CachedRuneLocale);
		CachedRuneLocale = xrl;
		XL_RETAIN(CachedRuneLocale);
		os_unfair_lock_unlock(&cache_lock);
	} else
		XL_RELEASE(xrl);

	return (ret);
}

#ifdef UNIFDEF_LEGACY_RUNE_APIS
int
setrunelocale(const char *encoding)
{
	int ret;

	XL_LOCK(&__global_locale);
	ret = __setrunelocale(encoding, &__global_locale);
	XL_UNLOCK(&__global_locale);
	return ret;
}
#endif /* UNIFDEF_LEGACY_RUNE_APIS */

__private_extern__ int
__wrap_setrunelocale(const char *locale, locale_t loc)
{
	int ret = __setrunelocale(locale, loc);

	if (ret != 0) {
		errno = ret;
		return (_LDP_ERROR);
	}
	return (_LDP_LOADED);
}

