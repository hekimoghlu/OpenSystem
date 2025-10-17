/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 13, 2022.
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
* Module m_sub                                                             *
* Menus subwindow association routines                                     *
***************************************************************************/

#include "menu.priv.h"

MODULE_ID("$Id: m_sub.c,v 1.12 2010/01/23 21:20:11 tom Exp $")

/*---------------------------------------------------------------------------
|   Facility      :  libnmenu  
|   Function      :  int set_menu_sub(MENU *menu, WINDOW *win)
|   
|   Description   :  Sets the subwindow of the menu.
|
|   Return Values :  E_OK           - success
|                    E_POSTED       - menu is already posted
+--------------------------------------------------------------------------*/
NCURSES_EXPORT(int)
set_menu_sub(MENU * menu, WINDOW *win)
{
  T((T_CALLED("set_menu_sub(%p,%p)"), (void *)menu, (void *)win));

  if (menu)
    {
      if (menu->status & _POSTED)
	RETURN(E_POSTED);
      else
#if NCURSES_SP_FUNCS
	{
	  /* We ensure that usersub is never null. So even if a null
	     WINDOW parameter is passed, we store the SCREENS stdscr.
	     The only MENU that can have a null usersub is the static
	     _nc_default_Menu.
	   */
	  SCREEN *sp = _nc_screen_of(menu->usersub);

	  menu->usersub = win ? win : sp->_stdscr;
	  _nc_Calculate_Item_Length_and_Width(menu);
	}
#else
	menu->usersub = win;
#endif
    }
  else
    _nc_Default_Menu.usersub = win;

  RETURN(E_OK);
}

/*---------------------------------------------------------------------------
|   Facility      :  libnmenu  
|   Function      :  WINDOW* menu_sub(const MENU *menu)
|   
|   Description   :  Returns a pointer to the subwindow of the menu
|
|   Return Values :  NULL on error, otherwise a pointer to the window
+--------------------------------------------------------------------------*/
NCURSES_EXPORT(WINDOW *)
menu_sub(const MENU * menu)
{
  const MENU *m = Normalize_Menu(menu);

  T((T_CALLED("menu_sub(%p)"), (const void *)menu));
  returnWin(Get_Menu_Window(m));
}

/* m_sub.c ends here */
