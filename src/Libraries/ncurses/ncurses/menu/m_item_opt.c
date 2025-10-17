/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 10, 2023.
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
* Module m_item_opt                                                        *
* Menus item option routines                                               *
***************************************************************************/

#include "menu.priv.h"

MODULE_ID("$Id: m_item_opt.c,v 1.18 2010/01/23 21:20:10 tom Exp $")

/*---------------------------------------------------------------------------
|   Facility      :  libnmenu  
|   Function      :  int set_item_opts(ITEM *item, Item_Options opts)  
|   
|   Description   :  Set the options of the item. If there are relevant
|                    changes, the item is connected and the menu is posted,
|                    the menu will be redisplayed.
|
|   Return Values :  E_OK            - success
|                    E_BAD_ARGUMENT  - invalid item options
+--------------------------------------------------------------------------*/
NCURSES_EXPORT(int)
set_item_opts(ITEM * item, Item_Options opts)
{
  T((T_CALLED("set_menu_opts(%p,%d)"), (void *)item, opts));

  opts &= ALL_ITEM_OPTS;

  if (opts & ~ALL_ITEM_OPTS)
    RETURN(E_BAD_ARGUMENT);

  if (item)
    {
      if (item->opt != opts)
	{
	  MENU *menu = item->imenu;

	  item->opt = opts;

	  if ((!(opts & O_SELECTABLE)) && item->value)
	    item->value = FALSE;

	  if (menu && (menu->status & _POSTED))
	    {
	      Move_And_Post_Item(menu, item);
	      _nc_Show_Menu(menu);
	    }
	}
    }
  else
    _nc_Default_Item.opt = opts;

  RETURN(E_OK);
}

/*---------------------------------------------------------------------------
|   Facility      :  libnmenu  
|   Function      :  int item_opts_off(ITEM *item, Item_Options opts)   
|   
|   Description   :  Switch of the options for this item.
|
|   Return Values :  E_OK            - success
|                    E_BAD_ARGUMENT  - invalid options
+--------------------------------------------------------------------------*/
NCURSES_EXPORT(int)
item_opts_off(ITEM * item, Item_Options opts)
{
  ITEM *citem = item;		/* use a copy because set_item_opts must detect

				   NULL item itself to adjust its behavior */

  T((T_CALLED("item_opts_off(%p,%d)"), (void *)item, opts));

  if (opts & ~ALL_ITEM_OPTS)
    RETURN(E_BAD_ARGUMENT);
  else
    {
      Normalize_Item(citem);
      opts = citem->opt & ~(opts & ALL_ITEM_OPTS);
      returnCode(set_item_opts(item, opts));
    }
}

/*---------------------------------------------------------------------------
|   Facility      :  libnmenu  
|   Function      :  int item_opts_on(ITEM *item, Item_Options opts)   
|   
|   Description   :  Switch on the options for this item.
|
|   Return Values :  E_OK            - success
|                    E_BAD_ARGUMENT  - invalid options
+--------------------------------------------------------------------------*/
NCURSES_EXPORT(int)
item_opts_on(ITEM * item, Item_Options opts)
{
  ITEM *citem = item;		/* use a copy because set_item_opts must detect

				   NULL item itself to adjust its behavior */

  T((T_CALLED("item_opts_on(%p,%d)"), (void *)item, opts));

  opts &= ALL_ITEM_OPTS;
  if (opts & ~ALL_ITEM_OPTS)
    RETURN(E_BAD_ARGUMENT);
  else
    {
      Normalize_Item(citem);
      opts = citem->opt | opts;
      returnCode(set_item_opts(item, opts));
    }
}

/*---------------------------------------------------------------------------
|   Facility      :  libnmenu  
|   Function      :  Item_Options item_opts(const ITEM *item)   
|   
|   Description   :  Switch of the options for this item.
|
|   Return Values :  Items options
+--------------------------------------------------------------------------*/
NCURSES_EXPORT(Item_Options)
item_opts(const ITEM * item)
{
  T((T_CALLED("item_opts(%p)"), (const void *)item));
  returnItemOpts(ALL_ITEM_OPTS & Normalize_Item(item)->opt);
}

/* m_item_opt.c ends here */
