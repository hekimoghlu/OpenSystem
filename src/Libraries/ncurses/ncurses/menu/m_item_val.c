/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 11, 2022.
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
* Module m_item_val                                                        *
* Set and get menus item values                                            *
***************************************************************************/

#include "menu.priv.h"

MODULE_ID("$Id: m_item_val.c,v 1.15 2010/01/23 21:20:10 tom Exp $")

/*---------------------------------------------------------------------------
|   Facility      :  libnmenu  
|   Function      :  int set_item_value(ITEM *item, int value)
|   
|   Description   :  Programmatically set the item's selection value. This is
|                    only allowed if the item is selectable at all and if
|                    it is not connected to a single-valued menu.
|                    If the item is connected to a posted menu, the menu
|                    will be redisplayed.  
|
|   Return Values :  E_OK              - success
|                    E_REQUEST_DENIED  - not selectable or single valued menu
+--------------------------------------------------------------------------*/
NCURSES_EXPORT(int)
set_item_value(ITEM * item, bool value)
{
  MENU *menu;

  T((T_CALLED("set_item_value(%p,%d)"), (void *)item, value));
  if (item)
    {
      menu = item->imenu;

      if ((!(item->opt & O_SELECTABLE)) ||
	  (menu && (menu->opt & O_ONEVALUE)))
	RETURN(E_REQUEST_DENIED);

      if (item->value ^ value)
	{
	  item->value = value ? TRUE : FALSE;
	  if (menu)
	    {
	      if (menu->status & _POSTED)
		{
		  Move_And_Post_Item(menu, item);
		  _nc_Show_Menu(menu);
		}
	    }
	}
    }
  else
    _nc_Default_Item.value = value;

  RETURN(E_OK);
}

/*---------------------------------------------------------------------------
|   Facility      :  libnmenu  
|   Function      :  bool item_value(const ITEM *item)
|   
|   Description   :  Return the selection value of the item
|
|   Return Values :  TRUE   - if item is selected
|                    FALSE  - if item is not selected
+--------------------------------------------------------------------------*/
NCURSES_EXPORT(bool)
item_value(const ITEM * item)
{
  T((T_CALLED("item_value(%p)"), (const void *)item));
  returnBool((Normalize_Item(item)->value) ? TRUE : FALSE);
}

/* m_item_val.c ends here */
