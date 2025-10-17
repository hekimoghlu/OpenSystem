/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 21, 2022.
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
__FBSDID("$FreeBSD: src/lib/libc/string/wcscoll.c,v 1.3 2004/04/07 09:47:56 tjr Exp $");

#include "xlocale_private.h"

#include <errno.h>
#include <stdlib.h>
#include <string.h>
#include <wchar.h>
#include "collate.h"

#define NOTFORWARD	(DIRECTIVE_BACKWARD | DIRECTIVE_POSITION)

int
wcscoll_l(const wchar_t *ws1, const wchar_t *ws2, locale_t loc)
{
	int sverrno;
	int len, len2, prim, prim2, sec, sec2, ret, ret2;
	const wchar_t *t, *t2;
	wchar_t *tt = NULL, *tt2 = NULL;
	wchar_t *tr = NULL, *tr2 = NULL;
	struct __collate_st_info *info;

	NORMALIZE_LOCALE(loc);
	if (loc->__collate_load_error)
		/*
		 * Locale has no special collating order or could not be
		 * loaded, do a fast binary comparison.
		 */
		return (wcscmp(ws1, ws2));

	info = &loc->__lc_collate->__info;
	len = len2 = 1;
	ret = ret2 = 0;

	if ((info->directive[0] & NOTFORWARD) ||
	    (info->directive[1] & NOTFORWARD) ||
	    (!(info->flags && COLLATE_SUBST_DUP) &&
	    (info->subst_count[0] > 0 || info->subst_count[1] > 0))) {
		int direc, pass;
		for(pass = 0; pass < info->directive_count; pass++) {
			direc = info->directive[pass];
			if (pass == 0 || !(info->flags & COLLATE_SUBST_DUP)) {
				free(tt);
				tt = __collate_substitute(ws1, pass, loc);
				free(tt2);
				tt2 = tt ? __collate_substitute(ws2, pass, loc) : NULL;
			}
			if (direc & DIRECTIVE_BACKWARD) {
				wchar_t *bp, *fp, c;
				tr = __collate_wcsdup(tt ? tt : ws1);
				bp = tr;
				fp = tr + wcslen(tr) - 1;
				while(bp < fp) {
					c = *bp;
					*bp++ = *fp;
					*fp-- = c;
				}
				tr2 = __collate_wcsdup(tt2 ? tt2 : ws2);
				bp = tr2;
				fp = tr2 + wcslen(tr2) - 1;
				while(bp < fp) {
					c = *bp;
					*bp++ = *fp;
					*fp-- = c;
				}
				t = (const wchar_t *)tr;
				t2 = (const wchar_t *)tr2;
			} else if (tt) {
				t = (const wchar_t *)tt;
				t2 = (const wchar_t *)tt2;
			} else {
				t = (const wchar_t *)ws1;
				t2 = (const wchar_t *)ws2;
			}
			if(direc & DIRECTIVE_POSITION) {
				while(*t && *t2) {
					prim = prim2 = 0;
					__collate_lookup_which(t, &len, &prim, pass, loc);
					if (prim <= 0) {
						if (prim < 0) {
							errno = EINVAL;
							ret = -1;
							goto end;
						}
						prim = COLLATE_MAX_PRIORITY;
					}
					__collate_lookup_which(t2, &len2, &prim2, pass, loc);
					if (prim2 <= 0) {
						if (prim2 < 0) {
							errno = EINVAL;
							ret = -1;
							goto end;
						}
						prim2 = COLLATE_MAX_PRIORITY;
					}
					if(prim != prim2) {
						ret = prim - prim2;
						goto end;
					}
					t += len;
					t2 += len2;
				}
			} else {
				while(*t && *t2) {
					prim = prim2 = 0;
					while(*t) {
						__collate_lookup_which(t, &len, &prim, pass, loc);
						if(prim > 0)
							break;
						if (prim < 0) {
							errno = EINVAL;
							ret = -1;
							goto end;
						}
						t += len;
					}
					while(*t2) {
						__collate_lookup_which(t2, &len2, &prim2, pass, loc);
						if(prim2 > 0)
							break;
						if (prim2 < 0) {
							errno = EINVAL;
							ret = -1;
							goto end;
						}
						t2 += len2;
					}
					if(!prim || !prim2)
						break;
					if(prim != prim2) {
						ret = prim - prim2;
						goto end;
					}
					t += len;
					t2 += len2;
				}
			}
			if(!*t) {
				if(*t2) {
					ret = -(int)*t2;
					goto end;
				}
			} else {
				ret = *t;
				goto end;
			}
		}
		ret = 0;
		goto end;
	}

	/* optimized common case: order_start forward;forward and duplicate
	 * (or no) substitute tables */
	tt = __collate_substitute(ws1, 0, loc);
	if (tt == NULL) {
		tt2 = NULL;
		t = (const wchar_t *)ws1;
		t2 = (const wchar_t *)ws2;
	} else {
		tt2 = __collate_substitute(ws2, 0, loc);
		t = (const wchar_t *)tt;
		t2 = (const wchar_t *)tt2;
	}
	while(*t && *t2) {
		prim = prim2 = 0;
		while(*t) {
			__collate_lookup_l(t, &len, &prim, &sec, loc);
			if (prim > 0)
				break;
			if (prim < 0) {
				errno = EINVAL;
				ret = -1;
				goto end;
			}
			t += len;
		}
		while(*t2) {
			__collate_lookup_l(t2, &len2, &prim2, &sec2, loc);
			if (prim2 > 0)
				break;
			if (prim2 < 0) {
				errno = EINVAL;
				ret = -1;
				goto end;
			}
			t2 += len2;
		}
		if(!prim || !prim2)
			break;
		if(prim != prim2) {
			ret = prim - prim2;
			goto end;
		}
		if(!ret2)
			ret2 = sec - sec2;
		t += len;
		t2 += len2;
	}
	if(!*t && *t2)
		ret = -(int)*t2;
	else if(*t && !*t2)
		ret = *t;
	else if(!*t && !*t2)
		ret = ret2;
  end:
	sverrno = errno;
	free(tt);
	free(tt2);
	free(tr);
	free(tr2);
	errno = sverrno;

	return ret;
}

int
wcscoll(const wchar_t *ws1, const wchar_t *ws2)
{
	return wcscoll_l(ws1, ws2, __current_locale());
}
