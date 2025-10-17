/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 27, 2024.
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
* Module m_format                                                          *
* Set and get maximum numbers of rows and columns in menus                 *
***************************************************************************/

#include "menu.priv.h"

MODULE_ID("$Id: m_format.c,v 1.18 2012/06/09 23:54:02 tom Exp $")

#define minimum(a,b) ((a)<(b) ? (a): (b))

/*---------------------------------------------------------------------------
|   Facility      :  libnmenu
|   Function      :  int set_menu_format(MENU *menu, int rows, int cols)
|
|   Description   :  Sets the maximum number of rows and columns of items
|                    that may be displayed at one time on a menu. If the
|                    menu contains more items than can be displayed at
|                    once, the menu will be scrollable.
|
|   Return Values :  E_OK                   - success
|                    E_BAD_ARGUMENT         - invalid values passed
|                    E_NOT_CONNECTED        - there are no items connected
|                    E_POSTED               - the menu is already posted
+--------------------------------------------------------------------------*/
NCURSES_EXPORT(int)
set_menu_format(MENU * menu, int rows, int cols)
{
  int total_rows, total_cols;

  T((T_CALLED("set_menu_format(%p,%d,%d)"), (void *)menu, rows, cols));

  if (rows < 0 || cols < 0)
    RETURN(E_BAD_ARGUMENT);

  if (menu)
    {
      if (menu->status & _POSTED)
	RETURN(E_POSTED);

      if (!(menu->items))
	RETURN(E_NOT_CONNECTED);

      if (rows == 0)
	rows = menu->frows;
      if (cols == 0)
	cols = menu->fcols;

      if (menu->pattern)
	Reset_Pattern(menu);

      menu->frows = (short)rows;
      menu->fcols = (short)cols;

      assert(rows > 0 && cols > 0);
      total_rows = (menu->nitems - 1) / cols + 1;
      total_cols = (menu->opt & O_ROWMAJOR) ?
	minimum(menu->nitems, cols) :
	(menu->nitems - 1) / total_rows + 1;

      menu->rows = (short)total_rows;
      menu->cols = (short)total_cols;
      menu->arows = (short)minimum(total_rows, rows);
      menu->toprow = 0;
      menu->curitem = *(menu->items);
      assert(menu->curitem);
      SetStatus(menu, _LINK_NEEDED);
      _nc_Calculate_Item_Length_and_Width(menu);
    }
  else
    {
      if (rows > 0)
	_nc_Default_Menu.frows = (short)rows;
      if (cols > 0)
	_nc_Default_Menu.fcols = (short)cols;
    }

  RETURN(E_OK);
}

/*---------------------------------------------------------------------------
|   Facility      :  libnmenu
|   Function      :  void menu_format(const MENU *menu, int *rows, int *cols)
|
|   Description   :  Returns the maximum number of rows and columns that may
|                    be displayed at one time on menu.
|
|   Return Values :  -
+--------------------------------------------------------------------------*/
NCURSES_EXPORT(void)
menu_format(const MENU * menu, int *rows, int *cols)
{
  if (rows)
    *rows = Normalize_Menu(menu)->frows;
  if (cols)
    *cols = Normalize_Menu(menu)->fcols;
}

/* m_format.c ends here */
