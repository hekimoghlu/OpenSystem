/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 14, 2025.
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
* Module m_opts                                                            *
* Menus option routines                                                    *
***************************************************************************/

#include "menu.priv.h"

MODULE_ID("$Id: m_opts.c,v 1.20 2010/01/23 21:20:10 tom Exp $")

/*---------------------------------------------------------------------------
|   Facility      :  libnmenu
|   Function      :  int set_menu_opts(MENU *menu, Menu_Options opts)
|
|   Description   :  Set the options for this menu. If the new settings
|                    end up in a change of the geometry of the menu, it
|                    will be recalculated. This operation is forbidden if
|                    the menu is already posted.
|
|   Return Values :  E_OK           - success
|                    E_BAD_ARGUMENT - invalid menu options
|                    E_POSTED       - menu is already posted
+--------------------------------------------------------------------------*/
NCURSES_EXPORT(int)
set_menu_opts(MENU * menu, Menu_Options opts)
{
  T((T_CALLED("set_menu_opts(%p,%d)"), (void *)menu, opts));

  opts &= ALL_MENU_OPTS;

  if (opts & ~ALL_MENU_OPTS)
    RETURN(E_BAD_ARGUMENT);

  if (menu)
    {
      if (menu->status & _POSTED)
	RETURN(E_POSTED);

      if ((opts & O_ROWMAJOR) != (menu->opt & O_ROWMAJOR))
	{
	  /* we need this only if the layout really changed ... */
	  if (menu->items && menu->items[0])
	    {
	      menu->toprow = 0;
	      menu->curitem = menu->items[0];
	      assert(menu->curitem);
	      set_menu_format(menu, menu->frows, menu->fcols);
	    }
	}

      menu->opt = opts;

      if (opts & O_ONEVALUE)
	{
	  ITEM **item;

	  if (((item = menu->items) != (ITEM **) 0))
	    for (; *item; item++)
	      (*item)->value = FALSE;
	}

      if (opts & O_SHOWDESC)	/* this also changes the geometry */
	_nc_Calculate_Item_Length_and_Width(menu);
    }
  else
    _nc_Default_Menu.opt = opts;

  RETURN(E_OK);
}

/*---------------------------------------------------------------------------
|   Facility      :  libnmenu
|   Function      :  int menu_opts_off(MENU *menu, Menu_Options opts)
|
|   Description   :  Switch off the options for this menu. If the new settings
|                    end up in a change of the geometry of the menu, it
|                    will be recalculated. This operation is forbidden if
|                    the menu is already posted.
|
|   Return Values :  E_OK           - success
|                    E_BAD_ARGUMENT - invalid options
|                    E_POSTED       - menu is already posted
+--------------------------------------------------------------------------*/
NCURSES_EXPORT(int)
menu_opts_off(MENU * menu, Menu_Options opts)
{
  MENU *cmenu = menu;		/* use a copy because set_menu_opts must detect

				   NULL menu itself to adjust its behavior */

  T((T_CALLED("menu_opts_off(%p,%d)"), (void *)menu, opts));

  opts &= ALL_MENU_OPTS;
  if (opts & ~ALL_MENU_OPTS)
    RETURN(E_BAD_ARGUMENT);
  else
    {
      Normalize_Menu(cmenu);
      opts = cmenu->opt & ~opts;
      returnCode(set_menu_opts(menu, opts));
    }
}

/*---------------------------------------------------------------------------
|   Facility      :  libnmenu
|   Function      :  int menu_opts_on(MENU *menu, Menu_Options opts)
|
|   Description   :  Switch on the options for this menu. If the new settings
|                    end up in a change of the geometry of the menu, it
|                    will be recalculated. This operation is forbidden if
|                    the menu is already posted.
|
|   Return Values :  E_OK           - success
|                    E_BAD_ARGUMENT - invalid menu options
|                    E_POSTED       - menu is already posted
+--------------------------------------------------------------------------*/
NCURSES_EXPORT(int)
menu_opts_on(MENU * menu, Menu_Options opts)
{
  MENU *cmenu = menu;		/* use a copy because set_menu_opts must detect

				   NULL menu itself to adjust its behavior */

  T((T_CALLED("menu_opts_on(%p,%d)"), (void *)menu, opts));

  opts &= ALL_MENU_OPTS;
  if (opts & ~ALL_MENU_OPTS)
    RETURN(E_BAD_ARGUMENT);
  else
    {
      Normalize_Menu(cmenu);
      opts = cmenu->opt | opts;
      returnCode(set_menu_opts(menu, opts));
    }
}

/*---------------------------------------------------------------------------
|   Facility      :  libnmenu
|   Function      :  Menu_Options menu_opts(const MENU *menu)
|
|   Description   :  Return the options for this menu.
|
|   Return Values :  Menu options
+--------------------------------------------------------------------------*/
NCURSES_EXPORT(Menu_Options)
menu_opts(const MENU * menu)
{
  T((T_CALLED("menu_opts(%p)"), (const void *)menu));
  returnMenuOpts(ALL_MENU_OPTS & Normalize_Menu(menu)->opt);
}

/* m_opts.c ends here */
