/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 12, 2024.
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
/****************************************************************************
 *  Author: Juergen Pfeifer                                                 *
 *     and: Thomas E. Dickey                                                *
 ****************************************************************************/

/*
 *	lib_slkset.c
 *      Set soft label text.
 */
#include <curses.priv.h>
#include <ctype.h>

#if USE_WIDEC_SUPPORT
#if HAVE_WCTYPE_H
#include <wctype.h>
#endif
#endif

MODULE_ID("$Id: lib_slkset.c,v 1.24 2012/12/08 23:09:25 tom Exp $")

NCURSES_EXPORT(int)
NCURSES_SP_NAME(slk_set) (NCURSES_SP_DCLx int i, const char *astr, int format)
{
    SLK *slk;
    int offset = 0;
    int numchrs;
    int numcols;
    int limit;
    const char *str = astr;
    const char *p;

    T((T_CALLED("slk_set(%p, %d, \"%s\", %d)"), (void *) SP_PARM, i, str, format));

    if (SP_PARM == 0
	|| (slk = SP_PARM->_slk) == 0
	|| i < 1
	|| i > slk->labcnt
	|| format < 0
	|| format > 2)
	returnCode(ERR);
    if (str == 0)
	str = "";
    --i;			/* Adjust numbering of labels */

    limit = MAX_SKEY_LEN(SP_PARM->slk_format);
    while (isspace(UChar(*str)))
	str++;			/* skip over leading spaces  */
    p = str;

#if USE_WIDEC_SUPPORT
    numcols = 0;
    while (*p != 0) {
	mbstate_t state;
	wchar_t wc;
	size_t need;

	init_mb(state);
	need = mbrtowc(0, p, strlen(p), &state);
	if (need == (size_t) -1)
	    break;
	mbrtowc(&wc, p, need, &state);
	if (!iswprint((wint_t) wc))
	    break;
	if (wcwidth(wc) + numcols > limit)
	    break;
	numcols += wcwidth(wc);
	p += need;
    }
    numchrs = (int) (p - str);
#else
    while (isprint(UChar(*p)))
	p++;			/* The first non-print stops */

    numcols = (int) (p - str);
    if (numcols > limit)
	numcols = limit;
    numchrs = numcols;
#endif

    FreeIfNeeded(slk->ent[i].ent_text);
    if ((slk->ent[i].ent_text = strdup(str)) == 0)
	returnCode(ERR);
    slk->ent[i].ent_text[numchrs] = '\0';

    if ((slk->ent[i].form_text = (char *) _nc_doalloc(slk->ent[i].form_text,
						      (size_t) (limit +
								numchrs + 1))
	) == 0)
	returnCode(ERR);

    switch (format) {
    case 0:			/* left-justified */
	offset = 0;
	break;
    case 1:			/* centered */
	offset = (limit - numcols) / 2;
	break;
    case 2:			/* right-justified */
	offset = limit - numcols;
	break;
    }
    if (offset <= 0)
	offset = 0;
    else
	memset(slk->ent[i].form_text, ' ', (size_t) offset);

    memcpy(slk->ent[i].form_text + offset,
	   slk->ent[i].ent_text,
	   (size_t) numchrs);

    if (offset < limit) {
	memset(slk->ent[i].form_text + offset + numchrs,
	       ' ',
	       (size_t) (limit - (offset + numcols)));
    }

    slk->ent[i].form_text[numchrs - numcols + limit] = 0;
    slk->ent[i].dirty = TRUE;
    returnCode(OK);
}

#if NCURSES_SP_FUNCS
NCURSES_EXPORT(int)
slk_set(int i, const char *astr, int format)
{
    return NCURSES_SP_NAME(slk_set) (CURRENT_SCREEN, i, astr, format);
}
#endif
