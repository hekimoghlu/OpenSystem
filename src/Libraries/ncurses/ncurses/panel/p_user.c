/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 21, 2023.
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

/* p_user.c
 * Set/Get panels user pointer 
 */
#include "panel.priv.h"

MODULE_ID("$Id: p_user.c,v 1.8 2010/01/23 23:18:35 tom Exp $")

NCURSES_EXPORT(int)
set_panel_userptr(PANEL * pan, NCURSES_CONST void *uptr)
{
  T((T_CALLED("set_panel_userptr(%p,%p)"), (void *)pan, (NCURSES_CONST void *)uptr));
  if (!pan)
    returnCode(ERR);
  pan->user = uptr;
  returnCode(OK);
}

NCURSES_EXPORT(NCURSES_CONST void *)
panel_userptr(const PANEL * pan)
{
  T((T_CALLED("panel_userptr(%p)"), (const void *)pan));
  returnCVoidPtr(pan ? pan->user : (NCURSES_CONST void *)0);
}
