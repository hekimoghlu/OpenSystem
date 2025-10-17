/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 5, 2022.
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
 *  Author: Juergen Pfeifer                        1997                     *
 *     and: Thomas E. Dickey                                                *
 ****************************************************************************/

/*
 * $Id: nc_panel.h,v 1.7 2009/07/04 18:20:02 tom Exp $
 *
 *	nc_panel.h
 *
 *	Headerfile to provide an interface for the panel layer into
 *      the SCREEN structure of the ncurses core.
 */

#ifndef NC_PANEL_H
#define NC_PANEL_H 1

#ifdef __cplusplus
extern "C"
{
#endif

#include <ncurses_dll.h>

  struct panel; /* Forward Declaration */

  struct panelhook {
    struct panel*   top_panel;
    struct panel*   bottom_panel;
    struct panel*   stdscr_pseudo_panel;
#if NO_LEAKS
    int (*destroy)(struct panel *);
#endif
  };

  struct screen;		/* Forward declaration */
/* Retrieve the panelhook of the current screen */
  extern NCURSES_EXPORT(struct panelhook*)
    _nc_panelhook (void);
#if NCURSES_SP_FUNCS
  extern NCURSES_EXPORT(struct panelhook *)
    NCURSES_SP_NAME(_nc_panelhook) (SCREEN *);
#endif

#ifdef __cplusplus
}
#endif

#endif				/* NC_PANEL_H */
