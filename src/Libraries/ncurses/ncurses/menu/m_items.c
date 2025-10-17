/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 27, 2022.
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
* Module m_items                                                           *
* Connect and disconnect items to and from menus                           *
***************************************************************************/

#include "menu.priv.h"

MODULE_ID("$Id: m_items.c,v 1.17 2010/01/23 21:20:10 tom Exp $")

/*---------------------------------------------------------------------------
|   Facility      :  libnmenu  
|   Function      :  int set_menu_items(MENU *menu, ITEM **items)
|   
|   Description   :  Sets the item pointer array connected to menu.
|
|   Return Values :  E_OK           - success
|                    E_POSTED       - menu is already posted
|                    E_CONNECTED    - one or more items are already connected
|                                     to another menu.
|                    E_BAD_ARGUMENT - An incorrect menu or item array was
|                                     passed to the function
+--------------------------------------------------------------------------*/
NCURSES_EXPORT(int)
set_menu_items(MENU * menu, ITEM ** items)
{
  T((T_CALLED("set_menu_items(%p,%p)"), (void *)menu, (void *)items));

  if (!menu || (items && !(*items)))
    RETURN(E_BAD_ARGUMENT);

  if (menu->status & _POSTED)
    RETURN(E_POSTED);

  if (menu->items)
    _nc_Disconnect_Items(menu);

  if (items)
    {
      if (!_nc_Connect_Items(menu, items))
	RETURN(E_CONNECTED);
    }

  menu->items = items;
  RETURN(E_OK);
}

/*---------------------------------------------------------------------------
|   Facility      :  libnmenu  
|   Function      :  ITEM **menu_items(const MENU *menu)
|   
|   Description   :  Returns a pointer to the item pointer array of the menu
|
|   Return Values :  NULL on error
+--------------------------------------------------------------------------*/
NCURSES_EXPORT(ITEM **)
menu_items(const MENU * menu)
{
  T((T_CALLED("menu_items(%p)"), (const void *)menu));
  returnItemPtr(menu ? menu->items : (ITEM **) 0);
}

/*---------------------------------------------------------------------------
|   Facility      :  libnmenu  
|   Function      :  int item_count(const MENU *menu)
|   
|   Description   :  Get the number of items connected to the menu. If the
|                    menu pointer is NULL we return -1.         
|
|   Return Values :  Number of items or -1 to indicate error.
+--------------------------------------------------------------------------*/
NCURSES_EXPORT(int)
item_count(const MENU * menu)
{
  T((T_CALLED("item_count(%p)"), (const void *)menu));
  returnCode(menu ? menu->nitems : -1);
}

/* m_items.c ends here */
