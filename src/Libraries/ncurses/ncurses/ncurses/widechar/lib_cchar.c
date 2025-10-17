/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 6, 2022.
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
**	lib_cchar.c
**
**	The routines setcchar() and getcchar().
**
*/

#include <curses.priv.h>

MODULE_ID("$Id: lib_cchar.c,v 1.27 2014/02/01 22:10:42 tom Exp $")

/* 
 * The SuSv2 description leaves some room for interpretation.  We'll assume wch
 * points to a string which is L'\0' terminated, contains at least one
 * character with strictly positive width, which must be the first, and
 * contains no characters of negative width.
 */
NCURSES_EXPORT(int)
setcchar(cchar_t *wcval,
	 const wchar_t *wch,
	 const attr_t attrs,
	 NCURSES_PAIRS_T color_pair,
	 const void *opts)
{
    unsigned i;
    unsigned len;
    int code = OK;

    TR(TRACE_CCALLS, (T_CALLED("setcchar(%p,%s,%lu,%d,%p)"),
		      (void *) wcval, _nc_viswbuf(wch),
		      (unsigned long) attrs, (int) color_pair, opts));

    if (opts != NULL
	|| wch == NULL
	|| ((len = (unsigned) wcslen(wch)) > 1 && wcwidth(wch[0]) < 0)) {
	code = ERR;
    } else {
	if (len > CCHARW_MAX)
	    len = CCHARW_MAX;

	/*
	 * If we have a following spacing-character, stop at that point.  We
	 * are only interested in adding non-spacing characters.
	 */
	for (i = 1; i < len; ++i) {
	    if (wcwidth(wch[i]) != 0) {
		len = i;
		break;
	    }
	}

	memset(wcval, 0, sizeof(*wcval));

	if (len != 0) {
	    SetAttr(*wcval, attrs);
	    SetPair(CHDEREF(wcval), color_pair);
	    memcpy(&wcval->chars, wch, len * sizeof(wchar_t));
	    TR(TRACE_CCALLS, ("copy %d wchars, first is %s", len,
			      _tracecchar_t(wcval)));
	}
    }

    TR(TRACE_CCALLS, (T_RETURN("%d"), code));
    return (code);
}

NCURSES_EXPORT(int)
getcchar(const cchar_t *wcval,
	 wchar_t *wch,
	 attr_t *attrs,
	 NCURSES_PAIRS_T *color_pair,
	 void *opts)
{
    wchar_t *wp;
    int len;
    int code = ERR;

    TR(TRACE_CCALLS, (T_CALLED("getcchar(%p,%p,%p,%p,%p)"),
		      (const void *) wcval,
		      (void *) wch,
		      (void *) attrs,
		      (void *) color_pair,
		      opts));

    if (opts == NULL && wcval != NULL) {
	len = ((wp = wmemchr(wcval->chars, L'\0', (size_t) CCHARW_MAX))
	       ? (int) (wp - wcval->chars)
	       : CCHARW_MAX);

	if (wch == NULL) {
	    /*
	     * If the value is a null, set the length to 1.
	     * If the value is not a null, return the length plus 1 for null.
	     */
	    code = (len < CCHARW_MAX) ? (len + 1) : CCHARW_MAX;
	} else if (attrs == 0 || color_pair == 0) {
	    code = ERR;
	} else if (len >= 0) {
	    *attrs = AttrOf(*wcval) & A_ATTRIBUTES;
	    *color_pair = (NCURSES_PAIRS_T) GetPair(*wcval);
	    wmemcpy(wch, wcval->chars, (size_t) len);
	    wch[len] = L'\0';
	    code = OK;
	}
    }

    TR(TRACE_CCALLS, (T_RETURN("%d"), code));
    return (code);
}
