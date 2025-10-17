/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 15, 2023.
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
 *     and: Juergen Pfeifer                         1997-1999,2009          *
 ****************************************************************************/

/* p_move.c
 * Move a panel to a new location
 */
#include "panel.priv.h"

MODULE_ID("$Id: p_move.c,v 1.11 2010/01/23 21:22:15 tom Exp $")

NCURSES_EXPORT(int)
move_panel(PANEL * pan, int starty, int startx)
{
  int rc = ERR;

  T((T_CALLED("move_panel(%p,%d,%d)"), (void *)pan, starty, startx));

  if (pan)
    {
      GetHook(pan);
      if (IS_LINKED(pan))
	{
	  Touchpan(pan);
	  PANEL_UPDATE(pan, (PANEL *) 0);
	}
      rc = mvwin(pan->win, starty, startx);
    }
  returnCode(rc);
}
