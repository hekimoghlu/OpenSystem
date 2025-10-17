/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 13, 2022.
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

/* p_below.c
 */
#include "panel.priv.h"

MODULE_ID("$Id: p_below.c,v 1.9 2012/03/10 23:43:41 tom Exp $")

#if NCURSES_SP_FUNCS
NCURSES_EXPORT(PANEL *)
ceiling_panel(SCREEN * sp)
{
  T((T_CALLED("ceiling_panel(%p)"), (void *)sp));
  if (sp)
    {
      struct panelhook *ph = NCURSES_SP_NAME(_nc_panelhook) (sp);

      /* if top and bottom are equal, we have no or only the pseudo panel */
      returnPanel(EMPTY_STACK()? (PANEL *) 0 : _nc_top_panel);
    }
  else
    {
      if (0 == CURRENT_SCREEN)
	returnPanel(0);
      else
	returnPanel(ceiling_panel(CURRENT_SCREEN));
    }
}
#endif

NCURSES_EXPORT(PANEL *)
panel_below(const PANEL * pan)
{
  PANEL *result;

  T((T_CALLED("panel_below(%p)"), (const void *)pan));
  if (pan)
    {
      GetHook(pan);
      /* we must not return the pseudo panel */
      result = Is_Pseudo(pan->below) ? (PANEL *) 0 : pan->below;
    }
  else
    {
#if NCURSES_SP_FUNCS
      result = ceiling_panel(CURRENT_SCREEN);
#else
      /* if top and bottom are equal, we have no or only the pseudo panel */
      result = EMPTY_STACK()? (PANEL *) 0 : _nc_top_panel;
#endif
    }
  returnPanel(result);
}
