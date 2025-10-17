/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 22, 2023.
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
 *   Author:  Thomas E. Dickey, 2011                                        *
 ****************************************************************************/

/*
    Version Control
    $Id: ncurses_compat.c,v 1.3 2015/08/06 23:09:10 tom Exp $
  --------------------------------------------------------------------------*/

/*
 * Provide compatibility with older versions of ncurses.
 */
#include <ncurses_cfg.h>

#if HAVE_INTTYPES_H
# include <inttypes.h>
#else
# if HAVE_STDINT_H
#  include <stdint.h>
# endif
#endif

#include <curses.h>

#if defined(NCURSES_VERSION_PATCH)

#if NCURSES_VERSION_PATCH < 20081122
extern bool has_mouse(void);
extern int _nc_has_mouse(void);

bool
has_mouse(void)
{
  return (bool)_nc_has_mouse();
}
#endif

/*
 * These are provided by lib_gen.c:
 */
#if NCURSES_VERSION_PATCH < 20070331
extern bool (is_keypad) (const WINDOW *);
extern bool (is_scrollok) (const WINDOW *);

bool
is_keypad(const WINDOW *win)
{
  return ((win)->_use_keypad);
}

bool
  (is_scrollok) (const WINDOW *win)
{
  return ((win)->_scroll);
}
#endif

#if NCURSES_VERSION_PATCH < 20060107
extern int (getbegx) (WINDOW *);
extern int (getbegy) (WINDOW *);
extern int (getcurx) (WINDOW *);
extern int (getcury) (WINDOW *);
extern int (getmaxx) (WINDOW *);
extern int (getmaxy) (WINDOW *);
extern int (getparx) (WINDOW *);
extern int (getpary) (WINDOW *);

int
  (getbegy) (WINDOW *win)
{
  return ((win) ? (win)->_begy : ERR);
}

int
  (getbegx) (WINDOW *win)
{
  return ((win) ? (win)->_begx : ERR);
}

int
  (getcury) (WINDOW *win)
{
  return ((win) ? (win)->_cury : ERR);
}

int
  (getcurx) (WINDOW *win)
{
  return ((win) ? (win)->_curx : ERR);
}

int
  (getmaxy) (WINDOW *win)
{
  return ((win) ? ((win)->_maxy + 1) : ERR);
}

int
  (getmaxx) (WINDOW *win)
{
  return ((win) ? ((win)->_maxx + 1) : ERR);
}

int
  (getpary) (WINDOW *win)
{
  return ((win) ? (win)->_pary : ERR);
}

int
  (getparx) (WINDOW *win)
{
  return ((win) ? (win)->_parx : ERR);
}
#endif

#endif
