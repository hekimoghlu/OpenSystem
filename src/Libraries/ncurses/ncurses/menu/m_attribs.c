/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 12, 2022.
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
* Module m_attribs                                                         *
* Control menus display attributes                                         *
***************************************************************************/

#include "menu.priv.h"

MODULE_ID("$Id: m_attribs.c,v 1.17 2012/03/10 23:43:41 tom Exp $")

/* Macro to redraw menu if it is posted and changed */
#define Refresh_Menu(menu) \
   if ( (menu) && ((menu)->status & _POSTED) )\
   {\
      _nc_Draw_Menu( menu );\
      _nc_Show_Menu( menu );\
   }

/* "Template" macro to generate a function to set a menus attribute */
#define GEN_MENU_ATTR_SET_FCT( name ) \
NCURSES_IMPEXP int NCURSES_API set_menu_ ## name (MENU* menu, chtype attr) \
{\
  T((T_CALLED("set_menu_" #name "(%p,%s)"), (void *) menu, _traceattr(attr))); \
   if (!(attr==A_NORMAL || (attr & A_ATTRIBUTES)==attr))\
      RETURN(E_BAD_ARGUMENT);\
   if (menu && ( menu -> name != attr))\
     {\
       (menu -> name) = attr;\
       Refresh_Menu(menu);\
     }\
   Normalize_Menu( menu ) -> name = attr;\
   RETURN(E_OK);\
}

/* "Template" macro to generate a function to get a menu's attribute */
#define GEN_MENU_ATTR_GET_FCT( name ) \
NCURSES_IMPEXP chtype NCURSES_API menu_ ## name (const MENU * menu)\
{\
   T((T_CALLED("menu_" #name "(%p)"), (const void *) menu));\
   returnAttr(Normalize_Menu( menu ) -> name);\
}

/*---------------------------------------------------------------------------
|   Facility      :  libnmenu  
|   Function      :  int set_menu_fore(MENU *menu, chtype attr)
|   
|   Description   :  Set the attribute for selectable items. In single-
|                    valued menus this is used to highlight the current
|                    item ((i.e. where the cursor is), in multi-valued
|                    menus this is used to highlight the selected items.
|
|   Return Values :  E_OK              - success
|                    E_BAD_ARGUMENT    - an invalid value has been passed   
+--------------------------------------------------------------------------*/
GEN_MENU_ATTR_SET_FCT(fore)

/*---------------------------------------------------------------------------
|   Facility      :  libnmenu  
|   Function      :  chtype menu_fore(const MENU* menu)
|   
|   Description   :  Return the attribute used for selectable items that
|                    are current (single-valued menu) or selected (multi-
|                    valued menu).   
|
|   Return Values :  Attribute value
+--------------------------------------------------------------------------*/
GEN_MENU_ATTR_GET_FCT(fore)

/*---------------------------------------------------------------------------
|   Facility      :  libnmenu  
|   Function      :  int set_menu_back(MENU *menu, chtype attr)
|   
|   Description   :  Set the attribute for selectable but not yet selected
|                    items.
|
|   Return Values :  E_OK             - success  
|                    E_BAD_ARGUMENT   - an invalid value has been passed
+--------------------------------------------------------------------------*/
GEN_MENU_ATTR_SET_FCT(back)

/*---------------------------------------------------------------------------
|   Facility      :  libnmenu  
|   Function      :  chtype menu_back(const MENU *menu)
|   
|   Description   :  Return the attribute used for selectable but not yet
|                    selected items. 
|
|   Return Values :  Attribute value
+--------------------------------------------------------------------------*/
GEN_MENU_ATTR_GET_FCT(back)

/*---------------------------------------------------------------------------
|   Facility      :  libnmenu  
|   Function      :  int set_menu_grey(MENU *menu, chtype attr)
|   
|   Description   :  Set the attribute for unselectable items.
|
|   Return Values :  E_OK             - success
|                    E_BAD_ARGUMENT   - an invalid value has been passed    
+--------------------------------------------------------------------------*/
GEN_MENU_ATTR_SET_FCT(grey)

/*---------------------------------------------------------------------------
|   Facility      :  libnmenu  
|   Function      :  chtype menu_grey(const MENU *menu)
|   
|   Description   :  Return the attribute used for non-selectable items
|
|   Return Values :  Attribute value
+--------------------------------------------------------------------------*/
GEN_MENU_ATTR_GET_FCT(grey)

/* m_attribs.c ends here */
