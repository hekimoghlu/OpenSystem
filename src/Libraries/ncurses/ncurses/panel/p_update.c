/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 24, 2025.
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

/* p_update.c
 * wnoutrefresh windows in an orderly fashion
 */
#include "panel.priv.h"

MODULE_ID("$Id: p_update.c,v 1.11 2010/01/23 21:22:16 tom Exp $")

NCURSES_EXPORT(void)
NCURSES_SP_NAME(update_panels) (NCURSES_SP_DCL0)
{
  PANEL *pan;

  T((T_CALLED("update_panels(%p)"), (void *)SP_PARM));
  dBug(("--> update_panels"));

  if (SP_PARM)
    {
      GetScreenHook(SP_PARM);

      pan = _nc_bottom_panel;
      while (pan && pan->above)
	{
	  PANEL_UPDATE(pan, pan->above);
	  pan = pan->above;
	}

      pan = _nc_bottom_panel;
      while (pan)
	{
	  Wnoutrefresh(pan);
	  pan = pan->above;
	}
    }

  returnVoid;
}

#if NCURSES_SP_FUNCS
NCURSES_EXPORT(void)
update_panels(void)
{
  NCURSES_SP_NAME(update_panels) (CURRENT_SCREEN);
}
#endif
