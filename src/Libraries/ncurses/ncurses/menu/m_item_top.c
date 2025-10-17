/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 23, 2022.
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
* Module m_item_top                                                        *
* Set and get top menus item                                               *
***************************************************************************/

#include "menu.priv.h"

MODULE_ID("$Id: m_item_top.c,v 1.11 2010/01/23 21:20:10 tom Exp $")

/*---------------------------------------------------------------------------
|   Facility      :  libnmenu  
|   Function      :  int set_top_row(MENU *menu, int row)
|   
|   Description   :  Makes the specified row the top row in the menu
|
|   Return Values :  E_OK             - success
|                    E_BAD_ARGUMENT   - not a menu pointer or invalid row
|                    E_NOT_CONNECTED  - there are no items for the menu
+--------------------------------------------------------------------------*/
NCURSES_EXPORT(int)
set_top_row(MENU * menu, int row)
{
  ITEM *item;

  T((T_CALLED("set_top_row(%p,%d)"), (void *)menu, row));

  if (menu)
    {
      if (menu->status & _IN_DRIVER)
	RETURN(E_BAD_STATE);
      if (menu->items == (ITEM **) 0)
	RETURN(E_NOT_CONNECTED);

      if ((row < 0) || (row > (menu->rows - menu->arows)))
	RETURN(E_BAD_ARGUMENT);
    }
  else
    RETURN(E_BAD_ARGUMENT);

  if (row != menu->toprow)
    {
      if (menu->status & _LINK_NEEDED)
	_nc_Link_Items(menu);

      item = menu->items[(menu->opt & O_ROWMAJOR) ? (row * menu->cols) : row];
      assert(menu->pattern);
      Reset_Pattern(menu);
      _nc_New_TopRow_and_CurrentItem(menu, row, item);
    }

  RETURN(E_OK);
}

/*---------------------------------------------------------------------------
|   Facility      :  libnmenu  
|   Function      :  int top_row(const MENU *)
|   
|   Description   :  Return the top row of the menu
|
|   Return Values :  The row number or ERR if there is no row
+--------------------------------------------------------------------------*/
NCURSES_EXPORT(int)
top_row(const MENU * menu)
{
  T((T_CALLED("top_row(%p)"), (const void *)menu));
  if (menu && menu->items && *(menu->items))
    {
      assert((menu->toprow >= 0) && (menu->toprow < menu->rows));
      returnCode(menu->toprow);
    }
  else
    returnCode(ERR);
}

/* m_item_top.c ends here */
