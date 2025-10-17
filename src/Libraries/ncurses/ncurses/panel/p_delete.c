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
 *     and: Juergen Pfeifer                         1999,2008               *
 ****************************************************************************/

/* p_delete.c
 * Remove a panel from stack, if in it, and free struct
 */
#include "panel.priv.h"

MODULE_ID("$Id: p_delete.c,v 1.10 2010/01/23 21:22:16 tom Exp $")

NCURSES_EXPORT(int)
del_panel(PANEL * pan)
{
  int err = OK;

  T((T_CALLED("del_panel(%p)"), (void *)pan));
  if (pan)
    {
      dBug(("--> del_panel %s", USER_PTR(pan->user)));
      {
	GetHook(pan);
	HIDE_PANEL(pan, err, OK);
	free((void *)pan);
      }
    }
  else
    err = ERR;

  returnCode(err);
}
