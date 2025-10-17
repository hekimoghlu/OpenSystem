/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 8, 2023.
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
 *     and: Juergen Pfeifer                         1997-1999,2008          *
 ****************************************************************************/

/* p_bottom.c
 * Place a panel on bottom of the stack; may already be in the stack 
 */
#include "panel.priv.h"

MODULE_ID("$Id: p_bottom.c,v 1.13 2010/01/23 21:22:16 tom Exp $")

NCURSES_EXPORT(int)
bottom_panel(PANEL * pan)
{
  int err = OK;

  T((T_CALLED("bottom_panel(%p)"), (void *)pan));
  if (pan)
    {
      GetHook(pan);
      if (!Is_Bottom(pan))
	{

	  dBug(("--> bottom_panel %s", USER_PTR(pan->user)));

	  HIDE_PANEL(pan, err, OK);
	  assert(_nc_bottom_panel == _nc_stdscr_pseudo_panel);

	  dStack("<lb%d>", 1, pan);

	  pan->below = _nc_bottom_panel;
	  pan->above = _nc_bottom_panel->above;
	  if (pan->above)
	    pan->above->below = pan;
	  _nc_bottom_panel->above = pan;

	  dStack("<lb%d>", 9, pan);
	}
    }
  else
    err = ERR;

  returnCode(err);
}
