/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 20, 2024.
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
 *                                                                          *
 ****************************************************************************/

#include <curses.priv.h>

MODULE_ID("$Id: lib_driver.c,v 1.6 2014/04/11 08:21:23 jpf Exp $")

typedef struct DriverEntry {
    const char *name;
    TERM_DRIVER *driver;
} DRIVER_ENTRY;

static DRIVER_ENTRY DriverTable[] =
{
#ifdef __MINGW32__
    {"win32console", &_nc_WIN_DRIVER},
#endif
    {"tinfo", &_nc_TINFO_DRIVER}	/* must be last */
};

NCURSES_EXPORT(int)
_nc_get_driver(TERMINAL_CONTROL_BLOCK * TCB, const char *name, int *errret)
{
    int code = ERR;
    size_t i;
    TERM_DRIVER *res = (TERM_DRIVER *) 0;
    TERM_DRIVER *use = 0;

    T((T_CALLED("_nc_get_driver(%p, %s, %p)"),
       (void *) TCB, NonNull(name), (void *) errret));

    assert(TCB != 0);

    for (i = 0; i < SIZEOF(DriverTable); i++) {
	res = DriverTable[i].driver;
	if (strcmp(DriverTable[i].name, res->td_name(TCB)) == 0) {
	    if (res->td_CanHandle(TCB, name, errret)) {
		use = res;
		break;
	    }
	}
    }
    if (use != 0) {
	TCB->drv = use;
	code = OK;
    }
    returnCode(code);
}

NCURSES_EXPORT(int)
NCURSES_SP_NAME(has_key) (SCREEN *sp, int keycode)
{
    T((T_CALLED("has_key(%p, %d)"), (void *) sp, keycode));
    returnCode(IsValidTIScreen(sp) ? CallDriver_1(sp, td_kyExist, keycode) : FALSE);
}

NCURSES_EXPORT(int)
has_key(int keycode)
{
    return NCURSES_SP_NAME(has_key) (CURRENT_SCREEN, keycode);
}

NCURSES_EXPORT(int)
NCURSES_SP_NAME(_nc_mcprint) (SCREEN *sp, char *data, int len)
{
    int code = ERR;

    if (0 != TerminalOf(sp))
	code = CallDriver_2(sp, td_print, data, len);
    return (code);
}

NCURSES_EXPORT(int)
mcprint(char *data, int len)
{
    return NCURSES_SP_NAME(_nc_mcprint) (CURRENT_SCREEN, data, len);
}

NCURSES_EXPORT(int)
NCURSES_SP_NAME(doupdate) (SCREEN *sp)
{
    int code = ERR;

    T((T_CALLED("doupdate(%p)"), (void *) sp));

    if (IsValidScreen(sp))
	code = CallDriver(sp, td_update);

    returnCode(code);
}

NCURSES_EXPORT(int)
doupdate(void)
{
    return NCURSES_SP_NAME(doupdate) (CURRENT_SCREEN);
}

NCURSES_EXPORT(int)
NCURSES_SP_NAME(mvcur) (SCREEN *sp, int yold, int xold, int ynew, int xnew)
{
    int code = ERR;
    TR(TRACE_CALLS | TRACE_MOVE, (T_CALLED("mvcur(%p,%d,%d,%d,%d)"),
				  (void *) sp, yold, xold, ynew, xnew));
    if (HasTerminal(sp)) {
	code = CallDriver_4(sp, td_hwcur, yold, xold, ynew, xnew);
    }
    returnCode(code);
}

NCURSES_EXPORT(int)
mvcur(int yold, int xold, int ynew, int xnew)
/* optimized cursor move from (yold, xold) to (ynew, xnew) */
{
    return NCURSES_SP_NAME(mvcur) (CURRENT_SCREEN, yold, xold, ynew, xnew);
}
