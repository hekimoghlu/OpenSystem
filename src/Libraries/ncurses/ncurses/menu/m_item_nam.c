/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 7, 2024.
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
* Module m_item_nam                                                        *
* Get menus item name and description                                      *
***************************************************************************/

#include "menu.priv.h"

MODULE_ID("$Id: m_item_nam.c,v 1.15 2010/01/23 21:20:10 tom Exp $")

/*---------------------------------------------------------------------------
|   Facility      :  libnmenu  
|   Function      :  char *item_name(const ITEM *item)
|   
|   Description   :  Return name of menu item
|
|   Return Values :  See above; returns NULL if item is invalid
+--------------------------------------------------------------------------*/
NCURSES_EXPORT(const char *)
item_name(const ITEM * item)
{
  T((T_CALLED("item_name(%p)"), (const void *)item));
  returnCPtr((item) ? item->name.str : (char *)0);
}

/*---------------------------------------------------------------------------
|   Facility      :  libnmenu  
|   Function      :  char *item_description(const ITEM *item)
|   
|   Description   :  Returns description of item
|
|   Return Values :  See above; Returns NULL if item is invalid
+--------------------------------------------------------------------------*/
NCURSES_EXPORT(const char *)
item_description(const ITEM * item)
{
  T((T_CALLED("item_description(%p)"), (const void *)item));
  returnCPtr((item) ? item->description.str : (char *)0);
}

/* m_item_nam.c ends here */
