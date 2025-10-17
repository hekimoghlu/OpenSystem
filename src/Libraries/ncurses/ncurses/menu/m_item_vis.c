/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 13, 2023.
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
* Module m_item_vis                                                        *
* Tell if menu item is visible                                             *
***************************************************************************/

#include "menu.priv.h"

MODULE_ID("$Id: m_item_vis.c,v 1.16 2010/01/23 21:20:10 tom Exp $")

/*---------------------------------------------------------------------------
|   Facility      :  libnmenu  
|   Function      :  bool item_visible(const ITEM *item)
|   
|   Description   :  A item is visible if it currently appears in the
|                    subwindow of a posted menu.
|
|   Return Values :  TRUE  if visible
|                    FALSE if invisible
+--------------------------------------------------------------------------*/
NCURSES_EXPORT(bool)
item_visible(const ITEM * item)
{
  MENU *menu;

  T((T_CALLED("item_visible(%p)"), (const void *)item));
  if (item &&
      (menu = item->imenu) &&
      (menu->status & _POSTED) &&
      ((menu->toprow + menu->arows) > (item->y)) &&
      (item->y >= menu->toprow))
    returnBool(TRUE);
  else
    returnBool(FALSE);
}

/* m_item_vis.c ends here */
