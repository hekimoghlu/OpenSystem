/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 30, 2023.
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
* Module m_hook                                                            *
* Assign application specific routines for automatic invocation by menus   *
***************************************************************************/

#include "menu.priv.h"

MODULE_ID("$Id: m_hook.c,v 1.16 2012/03/10 23:43:41 tom Exp $")

/* "Template" macro to generate function to set application specific hook */
#define GEN_HOOK_SET_FUNCTION( typ, name ) \
NCURSES_IMPEXP int NCURSES_API set_ ## typ ## _ ## name (MENU *menu, Menu_Hook func )\
{\
   T((T_CALLED("set_" #typ "_" #name "(%p,%p)"), (void *) menu, func));\
   (Normalize_Menu(menu) -> typ ## name = func );\
   RETURN(E_OK);\
}

/* "Template" macro to generate function to get application specific hook */
#define GEN_HOOK_GET_FUNCTION( typ, name ) \
NCURSES_IMPEXP Menu_Hook NCURSES_API typ ## _ ## name ( const MENU *menu )\
{\
   T((T_CALLED(#typ "_" #name "(%p)"), (const void *) menu));\
   returnMenuHook(Normalize_Menu(menu) -> typ ## name);\
}

/*---------------------------------------------------------------------------
|   Facility      :  libnmenu  
|   Function      :  int set_menu_init(MENU *menu, void (*f)(MENU *))
|   
|   Description   :  Set user-exit which is called when menu is posted
|                    or just after the top row changes.
|
|   Return Values :  E_OK               - success
+--------------------------------------------------------------------------*/
GEN_HOOK_SET_FUNCTION(menu, init)

/*---------------------------------------------------------------------------
|   Facility      :  libnmenu  
|   Function      :  void (*)(MENU *) menu_init(const MENU *menu)
|   
|   Description   :  Return address of user-exit function which is called
|                    when a menu is posted or just after the top row 
|                    changes.
|
|   Return Values :  Menu init function address or NULL
+--------------------------------------------------------------------------*/
GEN_HOOK_GET_FUNCTION(menu, init)

/*---------------------------------------------------------------------------
|   Facility      :  libnmenu  
|   Function      :  int set_menu_term (MENU *menu, void (*f)(MENU *))
|   
|   Description   :  Set user-exit which is called when menu is unposted
|                    or just before the top row changes.
|
|   Return Values :  E_OK               - success
+--------------------------------------------------------------------------*/
GEN_HOOK_SET_FUNCTION(menu, term)

/*---------------------------------------------------------------------------
|   Facility      :  libnmenu  
|   Function      :  void (*)(MENU *) menu_term(const MENU *menu)
|   
|   Description   :  Return address of user-exit function which is called
|                    when a menu is unposted or just before the top row 
|                    changes.
|
|   Return Values :  Menu finalization function address or NULL
+--------------------------------------------------------------------------*/
GEN_HOOK_GET_FUNCTION(menu, term)

/*---------------------------------------------------------------------------
|   Facility      :  libnmenu  
|   Function      :  int set_item_init (MENU *menu, void (*f)(MENU *))
|   
|   Description   :  Set user-exit which is called when menu is posted
|                    or just after the current item changes.
|
|   Return Values :  E_OK               - success
+--------------------------------------------------------------------------*/
GEN_HOOK_SET_FUNCTION(item, init)

/*---------------------------------------------------------------------------
|   Facility      :  libnmenu  
|   Function      :  void (*)(MENU *) item_init (const MENU *menu)
|   
|   Description   :  Return address of user-exit function which is called
|                    when a menu is posted or just after the current item 
|                    changes.
|
|   Return Values :  Item init function address or NULL
+--------------------------------------------------------------------------*/
GEN_HOOK_GET_FUNCTION(item, init)

/*---------------------------------------------------------------------------
|   Facility      :  libnmenu  
|   Function      :  int set_item_term (MENU *menu, void (*f)(MENU *))
|   
|   Description   :  Set user-exit which is called when menu is unposted
|                    or just before the current item changes.
|
|   Return Values :  E_OK               - success
+--------------------------------------------------------------------------*/
GEN_HOOK_SET_FUNCTION(item, term)

/*---------------------------------------------------------------------------
|   Facility      :  libnmenu  
|   Function      :  void (*)(MENU *) item_init (const MENU *menu)
|   
|   Description   :  Return address of user-exit function which is called
|                    when a menu is unposted or just before the current item 
|                    changes.
|
|   Return Values :  Item finalization function address or NULL
+--------------------------------------------------------------------------*/
GEN_HOOK_GET_FUNCTION(item, term)

/* m_hook.c ends here */
