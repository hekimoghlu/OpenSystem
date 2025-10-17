/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 13, 2023.
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
 *   Author:  Thomas E. Dickey                                              *
 ****************************************************************************/

#include "menu.priv.h"

MODULE_ID("$Id: m_trace.c,v 1.4 2010/01/23 21:20:10 tom Exp $")

NCURSES_EXPORT(ITEM *)
_nc_retrace_item(ITEM * code)
{
  T((T_RETURN("%p"), (void *)code));
  return code;
}

NCURSES_EXPORT(ITEM **)
_nc_retrace_item_ptr(ITEM ** code)
{
  T((T_RETURN("%p"), (void *)code));
  return code;
}

NCURSES_EXPORT(Item_Options)
_nc_retrace_item_opts(Item_Options code)
{
  T((T_RETURN("%d"), code));
  return code;
}

NCURSES_EXPORT(MENU *)
_nc_retrace_menu(MENU * code)
{
  T((T_RETURN("%p"), (void *)code));
  return code;
}

NCURSES_EXPORT(Menu_Hook)
_nc_retrace_menu_hook(Menu_Hook code)
{
  T((T_RETURN("%p"), code));
  return code;
}

NCURSES_EXPORT(Menu_Options)
_nc_retrace_menu_opts(Menu_Options code)
{
  T((T_RETURN("%d"), code));
  return code;
}
