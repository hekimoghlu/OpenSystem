/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 4, 2021.
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
* Module m_pad                                                             *
* Control menus padding character                                          *
***************************************************************************/

#include "menu.priv.h"

MODULE_ID("$Id: m_pad.c,v 1.13 2012/03/10 23:43:41 tom Exp $")

/* Macro to redraw menu if it is posted and changed */
#define Refresh_Menu(menu) \
   if ( (menu) && ((menu)->status & _POSTED) )\
   {\
      _nc_Draw_Menu( menu );\
      _nc_Show_Menu( menu ); \
   }

/*---------------------------------------------------------------------------
|   Facility      :  libnmenu  
|   Function      :  int set_menu_pad(MENU* menu, int pad)
|   
|   Description   :  Set the character to be used to separate the item name
|                    from its description. This must be a printable 
|                    character.
|
|   Return Values :  E_OK              - success
|                    E_BAD_ARGUMENT    - an invalid value has been passed
+--------------------------------------------------------------------------*/
NCURSES_EXPORT(int)
set_menu_pad(MENU * menu, int pad)
{
  bool do_refresh = (menu != (MENU *) 0);

  T((T_CALLED("set_menu_pad(%p,%d)"), (void *)menu, pad));

  if (!isprint(UChar(pad)))
    RETURN(E_BAD_ARGUMENT);

  Normalize_Menu(menu);
  menu->pad = (unsigned char)pad;

  if (do_refresh)
    Refresh_Menu(menu);

  RETURN(E_OK);
}

/*---------------------------------------------------------------------------
|   Facility      :  libnmenu  
|   Function      :  int menu_pad(const MENU *menu)
|   
|   Description   :  Return the value of the padding character
|
|   Return Values :  The pad character
+--------------------------------------------------------------------------*/
NCURSES_EXPORT(int)
menu_pad(const MENU * menu)
{
  T((T_CALLED("menu_pad(%p)"), (const void *)menu));
  returnCode(Normalize_Menu(menu)->pad);
}

/* m_pad.c ends here */
