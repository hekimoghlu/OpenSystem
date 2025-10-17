/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 24, 2022.
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
 *	comp_error.c -- Error message routines
 *
 */

#include <curses.priv.h>

#include <tic.h>

MODULE_ID("$Id: comp_error.c,v 1.36 2012/02/22 22:34:31 tom Exp $")

NCURSES_EXPORT_VAR(bool) _nc_suppress_warnings = FALSE;
NCURSES_EXPORT_VAR(int) _nc_curr_line = 0; /* current line # in input */
NCURSES_EXPORT_VAR(int) _nc_curr_col = 0; /* current column # in input */

#define SourceName	_nc_globals.comp_sourcename
#define TermType	_nc_globals.comp_termtype

NCURSES_EXPORT(const char *)
_nc_get_source(void)
{
    return SourceName;
}

NCURSES_EXPORT(void)
_nc_set_source(const char *const name)
{
    FreeIfNeeded(SourceName);
    SourceName = strdup(name);
}

NCURSES_EXPORT(void)
_nc_set_type(const char *const name)
{
    if (TermType == 0)
	TermType = typeMalloc(char, MAX_NAME_SIZE + 1);
    if (TermType != 0) {
	TermType[0] = '\0';
	if (name)
	    strncat(TermType, name, (size_t) MAX_NAME_SIZE);
    }
}

NCURSES_EXPORT(void)
_nc_get_type(char *name)
{
#if NO_LEAKS
    if (name == 0 && TermType != 0) {
	FreeAndNull(TermType);
	return;
    }
#endif
    if (name != 0)
	_nc_STRCPY(name, TermType != 0 ? TermType : "", MAX_NAME_SIZE);
}

static NCURSES_INLINE void
where_is_problem(void)
{
    fprintf(stderr, "\"%s\"", SourceName ? SourceName : "?");
    if (_nc_curr_line >= 0)
	fprintf(stderr, ", line %d", _nc_curr_line);
    if (_nc_curr_col >= 0)
	fprintf(stderr, ", col %d", _nc_curr_col);
    if (TermType != 0 && TermType[0] != '\0')
	fprintf(stderr, ", terminal '%s'", TermType);
    fputc(':', stderr);
    fputc(' ', stderr);
}

NCURSES_EXPORT(void)
_nc_warning(const char *const fmt,...)
{
    va_list argp;

    if (_nc_suppress_warnings)
	return;

    where_is_problem();
    va_start(argp, fmt);
    vfprintf(stderr, fmt, argp);
    fprintf(stderr, "\n");
    va_end(argp);
}

NCURSES_EXPORT(void)
_nc_err_abort(const char *const fmt,...)
{
    va_list argp;

    where_is_problem();
    va_start(argp, fmt);
    vfprintf(stderr, fmt, argp);
    fprintf(stderr, "\n");
    va_end(argp);
    exit(EXIT_FAILURE);
}

NCURSES_EXPORT(void)
_nc_syserr_abort(const char *const fmt,...)
{
    va_list argp;

    where_is_problem();
    va_start(argp, fmt);
    vfprintf(stderr, fmt, argp);
    fprintf(stderr, "\n");
    va_end(argp);

    /* If we're debugging, try to show where the problem occurred - this
     * will dump core.
     */
#if defined(TRACE) || !defined(NDEBUG)
    abort();
#else
    /* Dumping core in production code is not a good idea.
     */
    exit(EXIT_FAILURE);
#endif
}

#if NO_LEAKS
NCURSES_EXPORT(void)
_nc_comp_error_leaks(void)
{
    FreeAndNull(SourceName);
    FreeAndNull(TermType);
}
#endif
