/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 27, 2024.
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
 *     and: Thomas E. Dickey                        1996-2003               *
 ****************************************************************************/

/*
**	lib_initscr.c
**
**	The routines initscr(), and termname().
**
*/

#include <curses.priv.h>

#if HAVE_SYS_TERMIO_H
#include <sys/termio.h>		/* needed for ISC */
#endif

MODULE_ID("$Id: lib_initscr.c,v 1.40 2014/04/26 18:47:51 juergen Exp $")

NCURSES_EXPORT(WINDOW *)
initscr(void)
{
    WINDOW *result;

    NCURSES_CONST char *name;

    START_TRACE();
    T((T_CALLED("initscr()")));

    _nc_init_pthreads();
    _nc_lock_global(curses);

    /* Portable applications must not call initscr() more than once */
    if (!_nc_globals.init_screen) {
	_nc_globals.init_screen = TRUE;

#ifdef __APPLE__
	if (getenv("NCURSES_PRINT_ABI") != NULL) {
		fprintf(stderr, "ncurses: minimum supported ABI: %d.%d, maximum supported ABI: %d.%d\n",
		    NCURSES_ABI_MAJOR(NCURSES_DEFAULT_ABI), NCURSES_ABI_MINOR(NCURSES_DEFAULT_ABI),
		    NCURSES_ABI_MAJOR(NCURSES_MAX_ABI), NCURSES_ABI_MINOR(NCURSES_MAX_ABI));
	}
#endif

	if ((name = getenv("TERM")) == 0
	    || *name == '\0')
	    name = "unknown";
#ifdef __CYGWIN__
	/*
	 * 2002/9/21
	 * Work around a bug in Cygwin.  Full-screen subprocesses run from
	 * bash, in turn spawned from another full-screen process, will dump
	 * core when attempting to write to stdout.  Opening /dev/tty
	 * explicitly seems to fix the problem.
	 */
	if (NC_ISATTY(fileno(stdout))) {
	    FILE *fp = fopen("/dev/tty", "w");
	    if (fp != 0 && NC_ISATTY(fileno(fp))) {
		fclose(stdout);
		dup2(fileno(fp), STDOUT_FILENO);
		stdout = fdopen(STDOUT_FILENO, "w");
	    }
	}
#endif
	if (newterm(name, stdout, stdin) == 0) {
	    fprintf(stderr, "Error opening terminal: %s.\n", name);
	    exit(EXIT_FAILURE);
	}

	/* def_shell_mode - done in newterm/_nc_setupscreen */
#if NCURSES_SP_FUNCS
	NCURSES_SP_NAME(def_prog_mode) (CURRENT_SCREEN);
#else
	def_prog_mode();
#endif
    }
    result = stdscr;
    _nc_unlock_global(curses);

    returnWin(result);
}
