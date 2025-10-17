/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 22, 2022.
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

MODULE_ID("$Id: fld_move.c,v 1.11 2012/03/11 00:37:16 tom Exp $")

/*---------------------------------------------------------------------------
|   Facility      :  libnform  
|   Function      :  int move_field(FIELD *field,int frow, int fcol)
|   
|   Description   :  Moves the disconnected field to the new location in
|                    the forms subwindow.
|
|   Return Values :  E_OK            - success
|                    E_BAD_ARGUMENT  - invalid argument passed
|                    E_CONNECTED     - field is connected
+--------------------------------------------------------------------------*/
NCURSES_EXPORT(int)
move_field(FIELD *field, int frow, int fcol)
{
  T((T_CALLED("move_field(%p,%d,%d)"), (void *)field, frow, fcol));

  if (!field || (frow < 0) || (fcol < 0))
    RETURN(E_BAD_ARGUMENT);

  if (field->form)
    RETURN(E_CONNECTED);

  field->frow = (short) frow;
  field->fcol = (short) fcol;
  RETURN(E_OK);
}

/* fld_move.c ends here */
