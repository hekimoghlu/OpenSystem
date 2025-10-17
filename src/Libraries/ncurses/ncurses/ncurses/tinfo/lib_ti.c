/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 25, 2021.
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
 *  Author: Zeyd M. Ben-Halim <zmbenhal@netcom.com> 1992,1995               *
 *     and: Eric S. Raymond <esr@snark.thyrsus.com>                         *
 *     and: Thomas E. Dickey                        1996-on                 *
 ****************************************************************************/

#include <curses.priv.h>

#include <tic.h>

MODULE_ID("$Id: lib_ti.c,v 1.30 2013/06/08 16:55:05 tom Exp $")

#if 0
static bool
same_name(const char *a, const char *b)
{
    fprintf(stderr, "compare(%s,%s)\n", a, b);
    return !strcmp(a, b);
}
#else
#define same_name(a,b) !strcmp(a,b)
#endif

NCURSES_EXPORT(int)
NCURSES_SP_NAME(tigetflag) (NCURSES_SP_DCLx NCURSES_CONST char *str)
{
    int result = ABSENT_BOOLEAN;
    int j = -1;

    T((T_CALLED("tigetflag(%p, %s)"), (void *) SP_PARM, str));

    if (HasTInfoTerminal(SP_PARM)) {
	TERMTYPE *tp = &(TerminalOf(SP_PARM)->type);
	struct name_table_entry const *entry_ptr;

	entry_ptr = _nc_find_type_entry(str, BOOLEAN, FALSE);
	if (entry_ptr != 0) {
	    j = entry_ptr->nte_index;
	}
#if NCURSES_XNAMES
	else {
	    int i;
	    for_each_ext_boolean(i, tp) {
		const char *capname = ExtBoolname(tp, i, boolnames);
		if (same_name(str, capname)) {
		    j = i;
		    break;
		}
	    }
	}
#endif
	if (j >= 0) {
	    /* note: setupterm forces invalid booleans to false */
	    result = tp->Booleans[j];
	}
    }

    returnCode(result);
}

#if NCURSES_SP_FUNCS
NCURSES_EXPORT(int)
tigetflag(NCURSES_CONST char *str)
{
    return NCURSES_SP_NAME(tigetflag) (CURRENT_SCREEN, str);
}
#endif

NCURSES_EXPORT(int)
NCURSES_SP_NAME(tigetnum) (NCURSES_SP_DCLx NCURSES_CONST char *str)
{
    int j = -1;
    int result = CANCELLED_NUMERIC;	/* Solaris returns a -1 on error */

    T((T_CALLED("tigetnum(%p, %s)"), (void *) SP_PARM, str));

    if (HasTInfoTerminal(SP_PARM)) {
	TERMTYPE *tp = &(TerminalOf(SP_PARM)->type);
	struct name_table_entry const *entry_ptr;

	entry_ptr = _nc_find_type_entry(str, NUMBER, FALSE);
	if (entry_ptr != 0) {
	    j = entry_ptr->nte_index;
	}
#if NCURSES_XNAMES
	else {
	    int i;
	    for_each_ext_number(i, tp) {
		const char *capname = ExtNumname(tp, i, numnames);
		if (same_name(str, capname)) {
		    j = i;
		    break;
		}
	    }
	}
#endif
	if (j >= 0) {
	    if (VALID_NUMERIC(tp->Numbers[j]))
		result = tp->Numbers[j];
	    else
		result = ABSENT_NUMERIC;
	}
    }

    returnCode(result);
}

#if NCURSES_SP_FUNCS
NCURSES_EXPORT(int)
tigetnum(NCURSES_CONST char *str)
{
    return NCURSES_SP_NAME(tigetnum) (CURRENT_SCREEN, str);
}
#endif

NCURSES_EXPORT(char *)
NCURSES_SP_NAME(tigetstr) (NCURSES_SP_DCLx NCURSES_CONST char *str)
{
    char *result = CANCELLED_STRING;
    int j = -1;

    T((T_CALLED("tigetstr(%p, %s)"), (void *) SP_PARM, str));

    if (HasTInfoTerminal(SP_PARM)) {
	TERMTYPE *tp = &(TerminalOf(SP_PARM)->type);
	struct name_table_entry const *entry_ptr;

	entry_ptr = _nc_find_type_entry(str, STRING, FALSE);
	if (entry_ptr != 0) {
	    j = entry_ptr->nte_index;
	}
#if NCURSES_XNAMES
	else {
	    int i;
	    for_each_ext_string(i, tp) {
		const char *capname = ExtStrname(tp, i, strnames);
		if (same_name(str, capname)) {
		    j = i;
		    break;
		}
	    }
	}
#endif
	if (j >= 0) {
	    /* note: setupterm forces cancelled strings to null */
	    result = tp->Strings[j];
	}
    }

    returnPtr(result);
}

#if NCURSES_SP_FUNCS
NCURSES_EXPORT(char *)
tigetstr(NCURSES_CONST char *str)
{
    return NCURSES_SP_NAME(tigetstr) (CURRENT_SCREEN, str);
}
#endif
