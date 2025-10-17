/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 2, 2023.
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

/*
 * clear.c --  clears the terminal's screen
 */

#define USE_LIBTINFO
#include <progs.priv.h>

MODULE_ID("$Id: clear.c,v 1.13 2013/06/22 22:20:54 tom Exp $")

static int
putch(int c)
{
    return putchar(c);
}

int
main(
	int argc GCC_UNUSED,
	char *argv[]GCC_UNUSED)
{
    char *E3;

    setupterm((char *) 0, STDOUT_FILENO, (int *) 0);

    /* Clear the scrollback buffer if possible. */
    E3 = tigetstr("E3");
    if (E3)
	(void) tputs(E3, lines > 0 ? lines : 1, putch);

    ExitProgram((tputs(clear_screen, lines > 0 ? lines : 1, putch) == ERR)
		? EXIT_FAILURE
		: EXIT_SUCCESS);
}
