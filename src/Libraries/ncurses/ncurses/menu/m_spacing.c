/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 8, 2024.
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
 *   Author:  Juergen Pfeifer, 1995,1997                                    *
 ****************************************************************************/

/***************************************************************************
* Module m_spacing                                                         *
* Routines to handle spacing between entries                               *
***************************************************************************/

#include "menu.priv.h"

MODULE_ID("$Id: m_spacing.c,v 1.19 2012/03/10 23:43:41 tom Exp $")

#define MAX_SPC_DESC ((TABSIZE) ? (TABSIZE) : 8)
#define MAX_SPC_COLS ((TABSIZE) ? (TABSIZE) : 8)
#define MAX_SPC_ROWS (3)

/*---------------------------------------------------------------------------
|   Facility      :  libnmenu
|   Function      :  int set_menu_spacing(MENU *menu,int desc, int r, int c);
|
|   Description   :  Set the spacing between entries
|
|   Return Values :  E_OK                 - on success
+--------------------------------------------------------------------------*/
NCURSES_EXPORT(int)
set_menu_spacing(MENU * menu, int s_desc, int s_row, int s_col)
{
  MENU *m;			/* split for ATAC workaround */

  T((T_CALLED("set_menu_spacing(%p,%d,%d,%d)"),
     (void *)menu, s_desc, s_row, s_col));

  m = Normalize_Menu(menu);

  assert(m);
  if (m->status & _POSTED)
    RETURN(E_POSTED);

  if (((s_desc < 0) || (s_desc > MAX_SPC_DESC)) ||
      ((s_row < 0) || (s_row > MAX_SPC_ROWS)) ||
      ((s_col < 0) || (s_col > MAX_SPC_COLS)))
    RETURN(E_BAD_ARGUMENT);

  m->spc_desc = (short)(s_desc ? s_desc : 1);
  m->spc_rows = (short)(s_row ? s_row : 1);
  m->spc_cols = (short)(s_col ? s_col : 1);
  _nc_Calculate_Item_Length_and_Width(m);

  RETURN(E_OK);
}

/*---------------------------------------------------------------------------
|   Facility      :  libnmenu
|   Function      :  int menu_spacing (const MENU *,int *,int *,int *);
|
|   Description   :  Retrieve info about spacing between the entries
|
|   Return Values :  E_OK             - on success
+--------------------------------------------------------------------------*/
NCURSES_EXPORT(int)
menu_spacing(const MENU * menu, int *s_desc, int *s_row, int *s_col)
{
  const MENU *m;		/* split for ATAC workaround */

  T((T_CALLED("menu_spacing(%p,%p,%p,%p)"),
     (const void *)menu,
     (void *)s_desc,
     (void *)s_row,
     (void *)s_col));

  m = Normalize_Menu(menu);

  assert(m);
  if (s_desc)
    *s_desc = m->spc_desc;
  if (s_row)
    *s_row = m->spc_rows;
  if (s_col)
    *s_col = m->spc_cols;

  RETURN(E_OK);
}

/* m_spacing.c ends here */
