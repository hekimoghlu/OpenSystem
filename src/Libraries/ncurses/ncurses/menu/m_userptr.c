/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 19, 2024.
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
* Module m_userptr                                                         *
* Associate application data with menus                                    *
***************************************************************************/

#include "menu.priv.h"

MODULE_ID("$Id: m_userptr.c,v 1.17 2010/01/23 21:20:10 tom Exp $")

/*---------------------------------------------------------------------------
|   Facility      :  libnmenu  
|   Function      :  int set_menu_userptr(MENU *menu, void *userptr)
|   
|   Description   :  Set the pointer that is reserved in any menu to store
|                    application relevant informations.
|
|   Return Values :  E_OK         - success
+--------------------------------------------------------------------------*/
NCURSES_EXPORT(int)
set_menu_userptr(MENU * menu, void *userptr)
{
  T((T_CALLED("set_menu_userptr(%p,%p)"), (void *)menu, (void *)userptr));
  Normalize_Menu(menu)->userptr = userptr;
  RETURN(E_OK);
}

/*---------------------------------------------------------------------------
|   Facility      :  libnmenu  
|   Function      :  void *menu_userptr(const MENU *menu)
|   
|   Description   :  Return the pointer that is reserved in any menu to
|                    store application relevant informations.
|
|   Return Values :  Value of the pointer. If no such pointer has been set,
|                    NULL is returned
+--------------------------------------------------------------------------*/
NCURSES_EXPORT(void *)
menu_userptr(const MENU * menu)
{
  T((T_CALLED("menu_userptr(%p)"), (const void *)menu));
  returnVoidPtr(Normalize_Menu(menu)->userptr);
}

/* m_userptr.c ends here */
