/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 3, 2023.
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
* Module m_pattern                                                         *
* Pattern matching handling                                                *
***************************************************************************/

#include "menu.priv.h"

MODULE_ID("$Id: m_pattern.c,v 1.16 2010/01/23 21:20:10 tom Exp $")

/*---------------------------------------------------------------------------
|   Facility      :  libnmenu  
|   Function      :  char *menu_pattern(const MENU *menu)
|   
|   Description   :  Return the value of the pattern buffer.
|
|   Return Values :  NULL          - if there is no pattern buffer allocated
|                    EmptyString   - if there is a pattern buffer but no
|                                    pattern is stored
|                    PatternString - as expected
+--------------------------------------------------------------------------*/
NCURSES_EXPORT(char *)
menu_pattern(const MENU * menu)
{
  static char empty[] = "";

  T((T_CALLED("menu_pattern(%p)"), (const void *)menu));
  returnPtr(menu ? (menu->pattern ? menu->pattern : empty) : 0);
}

/*---------------------------------------------------------------------------
|   Facility      :  libnmenu  
|   Function      :  int set_menu_pattern(MENU *menu, const char *p)
|   
|   Description   :  Set the match pattern for a menu and position to the
|                    first item that matches.
|
|   Return Values :  E_OK              - success
|                    E_BAD_ARGUMENT    - invalid menu or pattern pointer
|                    E_BAD_STATE       - menu in user hook routine
|                    E_NOT_CONNECTED   - no items connected to menu
|                    E_NO_MATCH        - no item matches pattern
+--------------------------------------------------------------------------*/
NCURSES_EXPORT(int)
set_menu_pattern(MENU * menu, const char *p)
{
  ITEM *matchitem;
  int matchpos;

  T((T_CALLED("set_menu_pattern(%p,%s)"), (void *)menu, _nc_visbuf(p)));

  if (!menu || !p)
    RETURN(E_BAD_ARGUMENT);

  if (!(menu->items))
    RETURN(E_NOT_CONNECTED);

  if (menu->status & _IN_DRIVER)
    RETURN(E_BAD_STATE);

  Reset_Pattern(menu);

  if (!(*p))
    {
      pos_menu_cursor(menu);
      RETURN(E_OK);
    }

  if (menu->status & _LINK_NEEDED)
    _nc_Link_Items(menu);

  matchpos = menu->toprow;
  matchitem = menu->curitem;
  assert(matchitem);

  while (*p)
    {
      if (!isprint(UChar(*p)) ||
	  (_nc_Match_Next_Character_In_Item_Name(menu, *p, &matchitem) != E_OK))
	{
	  Reset_Pattern(menu);
	  pos_menu_cursor(menu);
	  RETURN(E_NO_MATCH);
	}
      p++;
    }

  /* This is reached if there was a match. So we position to the new item */
  Adjust_Current_Item(menu, matchpos, matchitem);
  RETURN(E_OK);
}

/* m_pattern.c ends here */
