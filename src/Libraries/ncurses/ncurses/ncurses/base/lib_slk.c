/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 11, 2022.
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
 *  Authors:                                                                *
 *          Gerhard Fuernkranz                      1993 (original)         *
 *          Zeyd M. Ben-Halim                       1992,1995 (sic)         *
 *          Eric S. Raymond                                                 *
 *          Juergen Pfeifer                         1996-on                 *
 *          Thomas E. Dickey                                                *
 ****************************************************************************/

/*
 *	lib_slk.c
 *	Soft key routines.
 */

#include <curses.priv.h>
#include <ctype.h>

#ifndef CUR
#define CUR SP_TERMTYPE
#endif

MODULE_ID("$Id: lib_slk.c,v 1.48 2011/03/05 21:21:52 tom Exp $")

#ifdef USE_TERM_DRIVER
#define NumLabels    InfoOf(SP_PARM).numlabels
#define NoColorVideo InfoOf(SP_PARM).nocolorvideo
#define LabelWidth   InfoOf(SP_PARM).labelwidth
#define LabelHeight  InfoOf(SP_PARM).labelheight
#else
#define NumLabels    num_labels
#define NoColorVideo no_color_video
#define LabelWidth   label_width
#define LabelHeight  label_height
#endif

/*
 * Free any memory related to soft labels, return an error.
 */
static int
slk_failed(NCURSES_SP_DCL0)
{
    if ((0 != SP_PARM) && SP_PARM->_slk) {
	FreeIfNeeded(SP_PARM->_slk->ent);
	free(SP_PARM->_slk);
	SP_PARM->_slk = (SLK *) 0;
    }
    return ERR;
}

NCURSES_EXPORT(int)
_nc_format_slks(NCURSES_SP_DCLx int cols)
{
    int gap, i, x;
    int max_length;

    if (!SP_PARM || !SP_PARM->_slk)
	return ERR;

    max_length = SP_PARM->_slk->maxlen;
    if (SP_PARM->slk_format >= 3) {	/* PC style */
	gap = (cols - 3 * (3 + 4 * max_length)) / 2;

	if (gap < 1)
	    gap = 1;

	for (i = x = 0; i < SP_PARM->_slk->maxlab; i++) {
	    SP_PARM->_slk->ent[i].ent_x = x;
	    x += max_length;
	    x += (i == 3 || i == 7) ? gap : 1;
	}
    } else {
	if (SP_PARM->slk_format == 2) {		/* 4-4 */
	    gap = cols - (int) (SP_PARM->_slk->maxlab * max_length) - 6;

	    if (gap < 1)
		gap = 1;
	    for (i = x = 0; i < SP_PARM->_slk->maxlab; i++) {
		SP_PARM->_slk->ent[i].ent_x = x;
		x += max_length;
		x += (i == 3) ? gap : 1;
	    }
	} else {
	    if (SP_PARM->slk_format == 1) {	/* 1 -> 3-2-3 */
		gap = (cols - (SP_PARM->_slk->maxlab * max_length) - 5)
		    / 2;

		if (gap < 1)
		    gap = 1;
		for (i = x = 0; i < SP_PARM->_slk->maxlab; i++) {
		    SP_PARM->_slk->ent[i].ent_x = x;
		    x += max_length;
		    x += (i == 2 || i == 4) ? gap : 1;
		}
	    } else {
		return slk_failed(NCURSES_SP_ARG);
	    }
	}
    }
    SP_PARM->_slk->dirty = TRUE;

    return OK;
}

/*
 * Initialize soft labels.
 * Called from newterm()
 */
NCURSES_EXPORT(int)
_nc_slk_initialize(WINDOW *stwin, int cols)
{
    int i;
    int res = OK;
    size_t max_length;
    SCREEN *sp;
    int numlab;

    T((T_CALLED("_nc_slk_initialize()")));

    assert(stwin);

    sp = _nc_screen_of(stwin);
    if (0 == sp)
	returnCode(ERR);

    assert(TerminalOf(SP_PARM));

    numlab = NumLabels;

    if (SP_PARM->_slk) {	/* we did this already, so simply return */
	returnCode(OK);
    } else if ((SP_PARM->_slk = typeCalloc(SLK, 1)) == 0)
	returnCode(ERR);

    if (!SP_PARM->slk_format)
	SP_PARM->slk_format = _nc_globals.slk_format;

    /*
     * If we use colors, vidputs() will suppress video attributes that conflict
     * with colors.  In that case, we're still guaranteed that "reverse" would
     * work.
     */
    if ((NoColorVideo & 1) == 0)
	SetAttr(SP_PARM->_slk->attr, A_STANDOUT);
    else
	SetAttr(SP_PARM->_slk->attr, A_REVERSE);

    SP_PARM->_slk->maxlab = (short) ((numlab > 0)
				     ? numlab
				     : MAX_SKEY(SP_PARM->slk_format));
    SP_PARM->_slk->maxlen = (short) ((numlab > 0)
				     ? LabelWidth * LabelHeight
				     : MAX_SKEY_LEN(SP_PARM->slk_format));
    SP_PARM->_slk->labcnt = (short) ((SP_PARM->_slk->maxlab < MAX_SKEY(SP_PARM->slk_format))
				     ? MAX_SKEY(SP_PARM->slk_format)
				     : SP_PARM->_slk->maxlab);

    if (SP_PARM->_slk->maxlen <= 0
	|| SP_PARM->_slk->labcnt <= 0
	|| (SP_PARM->_slk->ent = typeCalloc(slk_ent,
					    (size_t) SP_PARM->_slk->labcnt))
	== NULL)
	returnCode(slk_failed(NCURSES_SP_ARG));

    max_length = (size_t) SP_PARM->_slk->maxlen;
    for (i = 0; i < SP_PARM->_slk->labcnt; i++) {
	size_t used = max_length + 1;

	SP_PARM->_slk->ent[i].ent_text = (char *) _nc_doalloc(0, used);
	if (SP_PARM->_slk->ent[i].ent_text == 0)
	    returnCode(slk_failed(NCURSES_SP_ARG));
	memset(SP_PARM->_slk->ent[i].ent_text, 0, used);

	SP_PARM->_slk->ent[i].form_text = (char *) _nc_doalloc(0, used);
	if (SP_PARM->_slk->ent[i].form_text == 0)
	    returnCode(slk_failed(NCURSES_SP_ARG));

	if (used > 1) {
	    memset(SP_PARM->_slk->ent[i].form_text, ' ', used - 1);
	}
	SP_PARM->_slk->ent[i].form_text[used - 1] = '\0';

	SP_PARM->_slk->ent[i].visible = (char) (i < SP_PARM->_slk->maxlab);
    }

    res = _nc_format_slks(NCURSES_SP_ARGx cols);

    if ((SP_PARM->_slk->win = stwin) == NULL) {
	returnCode(slk_failed(NCURSES_SP_ARG));
    }

    /* We now reset the format so that the next newterm has again
     * per default no SLK keys and may call slk_init again to
     * define a new layout. (juergen 03-Mar-1999)
     */
    _nc_globals.slk_format = 0;
    returnCode(res);
}

/*
 * Restore the soft labels on the screen.
 */
NCURSES_EXPORT(int)
NCURSES_SP_NAME(slk_restore) (NCURSES_SP_DCL0)
{
    T((T_CALLED("slk_restore(%p)"), (void *) SP_PARM));

    if (0 == SP_PARM)
	returnCode(ERR);
    if (SP_PARM->_slk == NULL)
	returnCode(ERR);
    SP_PARM->_slk->hidden = FALSE;
    SP_PARM->_slk->dirty = TRUE;

    returnCode(NCURSES_SP_NAME(slk_refresh) (NCURSES_SP_ARG));
}

#if NCURSES_SP_FUNCS
NCURSES_EXPORT(int)
slk_restore(void)
{
    return NCURSES_SP_NAME(slk_restore) (CURRENT_SCREEN);
}
#endif
