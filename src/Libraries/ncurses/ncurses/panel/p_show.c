/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 20, 2025.
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
 ****************************************************************************/

/* p_show.c
 * Place a panel on top of the stack; may already be in the stack 
 */
#include "panel.priv.h"

MODULE_ID("$Id: p_show.c,v 1.13 2010/01/23 21:22:16 tom Exp $")

NCURSES_EXPORT(int)
show_panel(PANEL * pan)
{
  int err = ERR;

  T((T_CALLED("show_panel(%p)"), (void *)pan));

  if (pan)
    {
      GetHook(pan);

      if (Is_Top(pan))
	returnCode(OK);

      dBug(("--> show_panel %s", USER_PTR(pan->user)));

      HIDE_PANEL(pan, err, OK);

      dStack("<lt%d>", 1, pan);
      assert(_nc_bottom_panel == _nc_stdscr_pseudo_panel);

      _nc_top_panel->above = pan;
      pan->below = _nc_top_panel;
      pan->above = (PANEL *) 0;
      _nc_top_panel = pan;

      err = OK;

      dStack("<lt%d>", 9, pan);
    }
  returnCode(err);
}
