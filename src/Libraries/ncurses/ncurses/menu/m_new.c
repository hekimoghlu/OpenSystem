/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 9, 2023.
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
* Module m_new                                                             *
* Creation and destruction of new menus                                    *
***************************************************************************/

#include "menu.priv.h"

MODULE_ID("$Id: m_new.c,v 1.21 2010/01/23 21:20:11 tom Exp $")

/*---------------------------------------------------------------------------
|   Facility      :  libnmenu  
|   Function      :  MENU* _nc_new_menu(SCREEN*, ITEM **items)
|   
|   Description   :  Creates a new menu connected to the item pointer
|                    array items and returns a pointer to the new menu.
|                    The new menu is initialized with the values from the
|                    default menu.
|
|   Return Values :  NULL on error
+--------------------------------------------------------------------------*/
NCURSES_EXPORT(MENU *)
NCURSES_SP_NAME(new_menu) (NCURSES_SP_DCLx ITEM ** items)
{
  int err = E_SYSTEM_ERROR;
  MENU *menu = typeCalloc(MENU, 1);

  T((T_CALLED("new_menu(%p,%p)"), (void *)SP_PARM, (void *)items));
  if (menu)
    {
      *menu = _nc_Default_Menu;
      menu->status = 0;
      menu->rows = menu->frows;
      menu->cols = menu->fcols;
#if NCURSES_SP_FUNCS
      /* This ensures userwin and usersub are always non-null,
         so we can derive always the SCREEN that this menu is
         running on. */
      menu->userwin = SP_PARM->_stdscr;
      menu->usersub = SP_PARM->_stdscr;
#endif
      if (items && *items)
	{
	  if (!_nc_Connect_Items(menu, items))
	    {
	      err = E_NOT_CONNECTED;
	      free(menu);
	      menu = (MENU *) 0;
	    }
	  else
	    err = E_OK;
	}
    }

  if (!menu)
    SET_ERROR(err);

  returnMenu(menu);
}

/*---------------------------------------------------------------------------
|   Facility      :  libnmenu  
|   Function      :  MENU *new_menu(ITEM **items)
|   
|   Description   :  Creates a new menu connected to the item pointer
|                    array items and returns a pointer to the new menu.
|                    The new menu is initialized with the values from the
|                    default menu.
|
|   Return Values :  NULL on error
+--------------------------------------------------------------------------*/
#if NCURSES_SP_FUNCS
NCURSES_EXPORT(MENU *)
new_menu(ITEM ** items)
{
  return NCURSES_SP_NAME(new_menu) (CURRENT_SCREEN, items);
}
#endif

/*---------------------------------------------------------------------------
|   Facility      :  libnmenu  
|   Function      :  int free_menu(MENU *menu)  
|   
|   Description   :  Disconnects menu from its associated item pointer 
|                    array and frees the storage allocated for the menu.
|
|   Return Values :  E_OK               - success
|                    E_BAD_ARGUMENT     - Invalid menu pointer passed
|                    E_POSTED           - Menu is already posted
+--------------------------------------------------------------------------*/
NCURSES_EXPORT(int)
free_menu(MENU * menu)
{
  T((T_CALLED("free_menu(%p)"), (void *)menu));
  if (!menu)
    RETURN(E_BAD_ARGUMENT);

  if (menu->status & _POSTED)
    RETURN(E_POSTED);

  if (menu->items)
    _nc_Disconnect_Items(menu);

  if ((menu->status & _MARK_ALLOCATED) && menu->mark)
    free(menu->mark);

  free(menu);
  RETURN(E_OK);
}

/* m_new.c ends here */
