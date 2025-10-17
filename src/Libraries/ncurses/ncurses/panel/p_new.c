/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 26, 2024.
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
 *     and: Juergen Pfeifer                         1997-1999               *
 *     and: Thomas E. Dickey                        2000-on                 *
 ****************************************************************************/

/* p_new.c
 * Creation of a new panel 
 */
#include "panel.priv.h"

MODULE_ID("$Id: p_new.c,v 1.16 2010/01/23 21:22:16 tom Exp $")

#ifdef TRACE
static char *stdscr_id;
static char *new_id;
#endif

/*+-------------------------------------------------------------------------
  Get root (i.e. stdscr's) panel.
  Establish the pseudo panel for stdscr if necessary.
--------------------------------------------------------------------------*/
static PANEL *
root_panel(NCURSES_SP_DCL0)
{
#if NCURSES_SP_FUNCS
  struct panelhook *ph = NCURSES_SP_NAME(_nc_panelhook) (sp);

#elif NO_LEAKS
  struct panelhook *ph = _nc_panelhook();
#endif

  if (_nc_stdscr_pseudo_panel == (PANEL *) 0)
    {

      assert(SP_PARM && SP_PARM->_stdscr && !_nc_bottom_panel && !_nc_top_panel);
#if NO_LEAKS
      ph->destroy = del_panel;
#endif
      _nc_stdscr_pseudo_panel = typeMalloc(PANEL, 1);
      if (_nc_stdscr_pseudo_panel != 0)
	{
	  PANEL *pan = _nc_stdscr_pseudo_panel;
	  WINDOW *win = SP_PARM->_stdscr;

	  pan->win = win;
	  pan->below = (PANEL *) 0;
	  pan->above = (PANEL *) 0;
#ifdef TRACE
	  if (!stdscr_id)
	    stdscr_id = strdup("stdscr");
	  pan->user = stdscr_id;
#else
	  pan->user = (void *)0;
#endif
	  _nc_bottom_panel = _nc_top_panel = pan;
	}
    }
  return _nc_stdscr_pseudo_panel;
}

NCURSES_EXPORT(PANEL *)
new_panel(WINDOW *win)
{
  PANEL *pan = (PANEL *) 0;

  GetWindowHook(win);

  T((T_CALLED("new_panel(%p)"), (void *)win));

  if (!win)
    returnPanel(pan);

  if (!_nc_stdscr_pseudo_panel)
    (void)root_panel(NCURSES_SP_ARG);
  assert(_nc_stdscr_pseudo_panel);

  if (!(win->_flags & _ISPAD) && (pan = typeMalloc(PANEL, 1)))
    {
      pan->win = win;
      pan->above = (PANEL *) 0;
      pan->below = (PANEL *) 0;
#ifdef TRACE
      if (!new_id)
	new_id = strdup("new");
      pan->user = new_id;
#else
      pan->user = (char *)0;
#endif
      (void)show_panel(pan);
    }
  returnPanel(pan);
}
