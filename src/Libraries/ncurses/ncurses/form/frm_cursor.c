/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 13, 2023.
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

#include "form.priv.h"

MODULE_ID("$Id: frm_cursor.c,v 1.10 2010/01/23 21:14:36 tom Exp $")

/*---------------------------------------------------------------------------
|   Facility      :  libnform  
|   Function      :  int pos_form_cursor(FORM * form)
|   
|   Description   :  Moves the form window cursor to the location required
|                    by the form driver to resume form processing. This may
|                    be needed after the application calls a curses library
|                    I/O routine that modifies the cursor position.
|
|   Return Values :  E_OK                      - Success
|                    E_SYSTEM_ERROR            - System error.
|                    E_BAD_ARGUMENT            - Invalid form pointer
|                    E_NOT_POSTED              - Form is not posted
+--------------------------------------------------------------------------*/
NCURSES_EXPORT(int)
pos_form_cursor(FORM *form)
{
  int res;

  T((T_CALLED("pos_form_cursor(%p)"), (void *)form));

  if (!form)
    res = E_BAD_ARGUMENT;
  else
    {
      if (!(form->status & _POSTED))
	res = E_NOT_POSTED;
      else
	res = _nc_Position_Form_Cursor(form);
    }
  RETURN(res);
}

/* frm_cursor.c ends here */
