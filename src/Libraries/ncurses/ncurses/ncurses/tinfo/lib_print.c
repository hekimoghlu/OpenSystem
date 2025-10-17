/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 7, 2022.
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
 *     and: Juergen Pfeifer                                                 *
 ****************************************************************************/

#include <curses.priv.h>

#ifndef CUR
#define CUR SP_TERMTYPE
#endif

MODULE_ID("$Id: lib_print.c,v 1.23 2012/02/22 22:34:31 tom Exp $")

NCURSES_EXPORT(int)
NCURSES_SP_NAME(mcprint) (NCURSES_SP_DCLx char *data, int len)
/* ship binary character data to the printer via mc4/mc5/mc5p */
{
    int result;
    char *mybuf, *switchon;
    size_t onsize, offsize;
    size_t need;

    errno = 0;
    if (!HasTInfoTerminal(SP_PARM)
	|| len <= 0
	|| (!prtr_non && (!prtr_on || !prtr_off))) {
	errno = ENODEV;
	return (ERR);
    }

    if (prtr_non) {
	switchon = TIPARM_1(prtr_non, len);
	onsize = strlen(switchon);
	offsize = 0;
    } else {
	switchon = prtr_on;
	onsize = strlen(prtr_on);
	offsize = strlen(prtr_off);
    }

    need = onsize + (size_t) len + offsize;

    if (switchon == 0
	|| (mybuf = typeMalloc(char, need + 1)) == 0) {
	errno = ENOMEM;
	return (ERR);
    }

    _nc_STRCPY(mybuf, switchon, need);
    memcpy(mybuf + onsize, data, (size_t) len);
    if (offsize)
	_nc_STRCPY(mybuf + onsize + len, prtr_off, need);

    /*
     * We're relying on the atomicity of UNIX writes here.  The
     * danger is that output from a refresh() might get interspersed
     * with the printer data after the write call returns but before the
     * data has actually been shipped to the terminal.  If the write(2)
     * operation is truly atomic we're protected from this.
     */
    result = (int) write(TerminalOf(SP_PARM)->Filedes, mybuf, need);

    /*
     * By giving up our scheduler slot here we increase the odds that the
     * kernel will ship the contiguous clist items from the last write
     * immediately.
     */
#ifndef __MINGW32__
    (void) sleep(0);
#endif
    free(mybuf);
    return (result);
}

#if NCURSES_SP_FUNCS && !defined(USE_TERM_DRIVER)
NCURSES_EXPORT(int)
mcprint(char *data, int len)
{
    return NCURSES_SP_NAME(mcprint) (CURRENT_SCREEN, data, len);
}
#endif
