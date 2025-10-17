/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 25, 2023.
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

/* p_above.c
 */
#include "panel.priv.h"

MODULE_ID("$Id: p_above.c,v 1.9 2012/03/10 23:43:41 tom Exp $")

#if NCURSES_SP_FUNCS
NCURSES_EXPORT(PANEL *)
ground_panel(SCREEN * sp)
{
  T((T_CALLED("ground_panel(%p)"), (void *)sp));
  if (sp)
    {
      struct panelhook *ph = NCURSES_SP_NAME(_nc_panelhook) (sp);

      if (_nc_bottom_panel)	/* this is the pseudo panel */
	returnPanel(_nc_bottom_panel->above);
      else
	returnPanel(0);
    }
  else
    {
      if (0 == CURRENT_SCREEN)
	returnPanel(0);
      else
	returnPanel(ground_panel(CURRENT_SCREEN));
    }
}
#endif

NCURSES_EXPORT(PANEL *)
panel_above(const PANEL * pan)
{
  PANEL *result;

  T((T_CALLED("panel_above(%p)"), (const void *)pan));
  if (pan)
    result = pan->above;
  else
    {
#if NCURSES_SP_FUNCS
      result = ground_panel(CURRENT_SCREEN);
#else
      /* if top and bottom are equal, we have no or only the pseudo panel;
         if not, we return the panel above the pseudo panel */
      result = EMPTY_STACK()? (PANEL *) 0 : _nc_bottom_panel->above;
#endif
    }
  returnPanel(result);
}
