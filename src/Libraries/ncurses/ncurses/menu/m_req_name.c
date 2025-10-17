/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 21, 2022.
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
* Module m_request_name                                                    *
* Routines to handle external names of menu requests                       *
***************************************************************************/

#include "menu.priv.h"

MODULE_ID("$Id: m_req_name.c,v 1.23 2015/04/04 18:00:23 tom Exp $")

#define DATA(s) { s }

static const char request_names[MAX_MENU_COMMAND - MIN_MENU_COMMAND + 1][14] =
{
  DATA("LEFT_ITEM"),
  DATA("RIGHT_ITEM"),
  DATA("UP_ITEM"),
  DATA("DOWN_ITEM"),
  DATA("SCR_ULINE"),
  DATA("SCR_DLINE"),
  DATA("SCR_DPAGE"),
  DATA("SCR_UPAGE"),
  DATA("FIRST_ITEM"),
  DATA("LAST_ITEM"),
  DATA("NEXT_ITEM"),
  DATA("PREV_ITEM"),
  DATA("TOGGLE_ITEM"),
  DATA("CLEAR_PATTERN"),
  DATA("BACK_PATTERN"),
  DATA("NEXT_MATCH"),
  DATA("PREV_MATCH")
};

#define A_SIZE (sizeof(request_names)/sizeof(request_names[0]))

/*---------------------------------------------------------------------------
|   Facility      :  libnmenu  
|   Function      :  const char * menu_request_name (int request);
|   
|   Description   :  Get the external name of a menu request.
|
|   Return Values :  Pointer to name      - on success
|                    NULL                 - on invalid request code
+--------------------------------------------------------------------------*/
NCURSES_EXPORT(const char *)
menu_request_name(int request)
{
  T((T_CALLED("menu_request_name(%d)"), request));
  if ((request < MIN_MENU_COMMAND) || (request > MAX_MENU_COMMAND))
    {
      SET_ERROR(E_BAD_ARGUMENT);
      returnCPtr((const char *)0);
    }
  else
    returnCPtr(request_names[request - MIN_MENU_COMMAND]);
}

/*---------------------------------------------------------------------------
|   Facility      :  libnmenu  
|   Function      :  int menu_request_by_name (const char *str);
|   
|   Description   :  Search for a request with this name.
|
|   Return Values :  Request Id       - on success
|                    E_NO_MATCH       - request not found
+--------------------------------------------------------------------------*/
NCURSES_EXPORT(int)
menu_request_by_name(const char *str)
{
  /* because the table is so small, it doesn't really hurt
     to run sequentially through it.
   */
  size_t i = 0;
  char buf[16];

  T((T_CALLED("menu_request_by_name(%s)"), _nc_visbuf(str)));

  if (str != 0 && (i = strlen(str)) != 0)
    {
      if (i > sizeof(buf) - 2)
	i = sizeof(buf) - 2;
      memcpy(buf, str, i);
      buf[i] = '\0';

      for (i = 0; buf[i] != '\0'; ++i)
	{
	  buf[i] = (char)toupper(UChar(buf[i]));
	}

      for (i = 0; i < A_SIZE; i++)
	{
	  if (strcmp(request_names[i], buf) == 0)
	    returnCode(MIN_MENU_COMMAND + (int)i);
	}
    }
  RETURN(E_NO_MATCH);
}

/* m_req_name.c ends here */
