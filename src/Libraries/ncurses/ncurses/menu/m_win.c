/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 2, 2024.
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
* Module m_win                                                             *
* Menus window association routines                                        *
***************************************************************************/

#include "menu.priv.h"

MODULE_ID("$Id: m_win.c,v 1.17 2010/01/23 21:20:11 tom Exp $")

/*---------------------------------------------------------------------------
|   Facility      :  libnmenu  
|   Function      :  int set_menu_win(MENU *menu, WINDOW *win)
|   
|   Description   :  Sets the window of the menu.
|
|   Return Values :  E_OK               - success
|                    E_POSTED           - menu is already posted
+--------------------------------------------------------------------------*/
NCURSES_EXPORT(int)
set_menu_win(MENU * menu, WINDOW *win)
{
  T((T_CALLED("set_menu_win(%p,%p)"), (void *)menu, (void *)win));

  if (menu)
    {
      if (menu->status & _POSTED)
	RETURN(E_POSTED);
      else
#if NCURSES_SP_FUNCS
	{
	  /* We ensure that userwin is never null. So even if a null
	     WINDOW parameter is passed, we store the SCREENS stdscr.
	     The only MENU that can have a null userwin is the static
	     _nc_default_Menu.
	   */
	  SCREEN *sp = _nc_screen_of(menu->userwin);

	  menu->userwin = win ? win : sp->_stdscr;
	  _nc_Calculate_Item_Length_and_Width(menu);
	}
#else
	menu->userwin = win;
#endif
    }
  else
    _nc_Default_Menu.userwin = win;

  RETURN(E_OK);
}

/*---------------------------------------------------------------------------
|   Facility      :  libnmenu  
|   Function      :  WINDOW* menu_win(const MENU*)
|   
|   Description   :  Returns pointer to the window of the menu
|
|   Return Values :  NULL on error, otherwise pointer to window
+--------------------------------------------------------------------------*/
NCURSES_EXPORT(WINDOW *)
menu_win(const MENU * menu)
{
  const MENU *m = Normalize_Menu(menu);

  T((T_CALLED("menu_win(%p)"), (const void *)menu));
  returnWin(Get_Menu_UserWin(m));
}

/* m_win.c ends here */
