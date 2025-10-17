/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 30, 2023.
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
* Module m_item_cur                                                        *
* Set and get current menus item                                           *
***************************************************************************/

#include "menu.priv.h"

MODULE_ID("$Id: m_item_cur.c,v 1.18 2010/01/23 21:20:10 tom Exp $")

/*---------------------------------------------------------------------------
|   Facility      :  libnmenu  
|   Function      :  int set_current_item(MENU *menu, const ITEM *item)
|   
|   Description   :  Make the item the current item
|
|   Return Values :  E_OK                - success
+--------------------------------------------------------------------------*/
NCURSES_EXPORT(int)
set_current_item(MENU * menu, ITEM * item)
{
  T((T_CALLED("set_current_item(%p,%p)"), (void *)menu, (void *)item));

  if (menu && item && (item->imenu == menu))
    {
      if (menu->status & _IN_DRIVER)
	RETURN(E_BAD_STATE);

      assert(menu->curitem);
      if (item != menu->curitem)
	{
	  if (menu->status & _LINK_NEEDED)
	    {
	      /*
	       * Items are available, but they are not linked together.
	       * So we have to link here.
	       */
	      _nc_Link_Items(menu);
	    }
	  assert(menu->pattern);
	  Reset_Pattern(menu);
	  /* adjust the window to make item visible and update the menu */
	  Adjust_Current_Item(menu, menu->toprow, item);
	}
    }
  else
    RETURN(E_BAD_ARGUMENT);

  RETURN(E_OK);
}

/*---------------------------------------------------------------------------
|   Facility      :  libnmenu  
|   Function      :  ITEM *current_item(const MENU *menu)
|   
|   Description   :  Return the menus current item
|
|   Return Values :  Item pointer or NULL if failure
+--------------------------------------------------------------------------*/
NCURSES_EXPORT(ITEM *)
current_item(const MENU * menu)
{
  T((T_CALLED("current_item(%p)"), (const void *)menu));
  returnItem((menu && menu->items) ? menu->curitem : (ITEM *) 0);
}

/*---------------------------------------------------------------------------
|   Facility      :  libnmenu  
|   Function      :  int item_index(const ITEM *)
|   
|   Description   :  Return the logical index of this item.
|
|   Return Values :  The index or ERR if this is an invalid item pointer
+--------------------------------------------------------------------------*/
NCURSES_EXPORT(int)
item_index(const ITEM * item)
{
  T((T_CALLED("item_index(%p)"), (const void *)item));
  returnCode((item && item->imenu) ? item->index : ERR);
}

/* m_item_cur.c ends here */
