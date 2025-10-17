/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 25, 2022.
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
* Module m_item_use                                                        *
* Associate application data with menu items                               *
***************************************************************************/

#include "menu.priv.h"

MODULE_ID("$Id: m_item_use.c,v 1.17 2010/01/23 21:20:10 tom Exp $")

/*---------------------------------------------------------------------------
|   Facility      :  libnmenu  
|   Function      :  int set_item_userptr(ITEM *item, void *userptr)
|   
|   Description   :  Set the pointer that is reserved in any item to store
|                    application relevant informations.  
|
|   Return Values :  E_OK               - success
+--------------------------------------------------------------------------*/
NCURSES_EXPORT(int)
set_item_userptr(ITEM * item, void *userptr)
{
  T((T_CALLED("set_item_userptr(%p,%p)"), (void *)item, (void *)userptr));
  Normalize_Item(item)->userptr = userptr;
  RETURN(E_OK);
}

/*---------------------------------------------------------------------------
|   Facility      :  libnmenu  
|   Function      :  void *item_userptr(const ITEM *item)
|   
|   Description   :  Return the pointer that is reserved in any item to store
|                    application relevant informations.
|
|   Return Values :  Value of the pointer. If no such pointer has been set,
|                    NULL is returned.
+--------------------------------------------------------------------------*/
NCURSES_EXPORT(void *)
item_userptr(const ITEM * item)
{
  T((T_CALLED("item_userptr(%p)"), (const void *)item));
  returnVoidPtr(Normalize_Item(item)->userptr);
}

/* m_item_use.c */
