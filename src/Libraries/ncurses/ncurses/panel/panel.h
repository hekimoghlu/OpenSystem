/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 10, 2023.
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
 *  Author: Zeyd M. Ben-Halim <zmbenhal@netcom.com> 1995                    *
 *     and: Eric S. Raymond <esr@snark.thyrsus.com>                         *
 *     and: Juergen Pfeifer                         1996-1999,2008          *
 ****************************************************************************/

/* $Id: panel.h,v 1.11 2009/04/11 19:50:40 tom Exp $ */

/* panel.h -- interface file for panels library */

#ifndef NCURSES_PANEL_H_incl
#define NCURSES_PANEL_H_incl 1

#include <curses.h>

typedef struct panel
{
  WINDOW *win;
  struct panel *below;
  struct panel *above;
  NCURSES_CONST void *user;
} PANEL;

#if	defined(__cplusplus)
extern "C" {
#endif

extern NCURSES_EXPORT(WINDOW*) panel_window (const PANEL *);
extern NCURSES_EXPORT(void)    update_panels (void);
extern NCURSES_EXPORT(int)     hide_panel (PANEL *);
extern NCURSES_EXPORT(int)     show_panel (PANEL *);
extern NCURSES_EXPORT(int)     del_panel (PANEL *);
extern NCURSES_EXPORT(int)     top_panel (PANEL *);
extern NCURSES_EXPORT(int)     bottom_panel (PANEL *);
extern NCURSES_EXPORT(PANEL*)  new_panel (WINDOW *);
extern NCURSES_EXPORT(PANEL*)  panel_above (const PANEL *);
extern NCURSES_EXPORT(PANEL*)  panel_below (const PANEL *);
extern NCURSES_EXPORT(int)     set_panel_userptr (PANEL *, NCURSES_CONST void *);
extern NCURSES_EXPORT(NCURSES_CONST void*) panel_userptr (const PANEL *);
extern NCURSES_EXPORT(int)     move_panel (PANEL *, int, int);
extern NCURSES_EXPORT(int)     replace_panel (PANEL *,WINDOW *);
extern NCURSES_EXPORT(int)     panel_hidden (const PANEL *);

#if NCURSES_SP_FUNCS
extern NCURSES_EXPORT(PANEL *) ground_panel(SCREEN *);
extern NCURSES_EXPORT(PANEL *) ceiling_panel(SCREEN *);

extern NCURSES_EXPORT(void)    NCURSES_SP_NAME(update_panels) (SCREEN*);
#endif

#if	defined(__cplusplus)
}
#endif

#endif /* NCURSES_PANEL_H_incl */

/* end of panel.h */
