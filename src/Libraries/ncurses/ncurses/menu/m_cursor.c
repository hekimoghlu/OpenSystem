/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 26, 2023.
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
* Module m_cursor                                                          *
* Correctly position a menu's cursor                                       *
***************************************************************************/

#include "menu.priv.h"

MODULE_ID("$Id: m_cursor.c,v 1.22 2010/01/23 21:20:10 tom Exp $")

/*---------------------------------------------------------------------------
|   Facility      :  libnmenu
|   Function      :  _nc_menu_cursor_pos
|
|   Description   :  Return position of logical cursor to current item
|
|   Return Values :  E_OK            - success
|                    E_BAD_ARGUMENT  - invalid menu
|                    E_NOT_POSTED    - Menu is not posted
+--------------------------------------------------------------------------*/
NCURSES_EXPORT(int)
_nc_menu_cursor_pos(const MENU * menu, const ITEM * item, int *pY, int *pX)
{
  if (!menu || !pX || !pY)
    return (E_BAD_ARGUMENT);
  else
    {
      if ((ITEM *) 0 == item)
	item = menu->curitem;
      assert(item != (ITEM *) 0);

      if (!(menu->status & _POSTED))
	return (E_NOT_POSTED);

      *pX = item->x * (menu->spc_cols + menu->itemlen);
      *pY = (item->y - menu->toprow) * menu->spc_rows;
    }
  return (E_OK);
}

/*---------------------------------------------------------------------------
|   Facility      :  libnmenu
|   Function      :  pos_menu_cursor
|
|   Description   :  Position logical cursor to current item in menu
|
|   Return Values :  E_OK            - success
|                    E_BAD_ARGUMENT  - invalid menu
|                    E_NOT_POSTED    - Menu is not posted
+--------------------------------------------------------------------------*/
NCURSES_EXPORT(int)
pos_menu_cursor(const MENU * menu)
{
  WINDOW *win, *sub;
  int x = 0, y = 0;
  int err = _nc_menu_cursor_pos(menu, (ITEM *) 0, &y, &x);

  T((T_CALLED("pos_menu_cursor(%p)"), (const void *)menu));

  if (E_OK == err)
    {
      win = Get_Menu_UserWin(menu);
      sub = menu->usersub ? menu->usersub : win;
      assert(win && sub);

      if ((menu->opt & O_SHOWMATCH) && (menu->pindex > 0))
	x += (menu->pindex + menu->marklen - 1);

      wmove(sub, y, x);

      if (win != sub)
	{
	  wcursyncup(sub);
	  wsyncup(sub);
	  untouchwin(sub);
	}
    }
  RETURN(err);
}

/* m_cursor.c ends here */
