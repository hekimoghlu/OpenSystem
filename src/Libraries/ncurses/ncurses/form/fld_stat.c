/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 1, 2023.
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

MODULE_ID("$Id: fld_stat.c,v 1.14 2012/06/10 00:13:09 tom Exp $")

/*---------------------------------------------------------------------------
|   Facility      :  libnform  
|   Function      :  int set_field_status(FIELD *field, bool status)
|   
|   Description   :  Set or clear the 'changed' indication flag for that
|                    fields primary buffer.
|
|   Return Values :  E_OK            - success
+--------------------------------------------------------------------------*/
NCURSES_EXPORT(int)
set_field_status(FIELD *field, bool status)
{
  T((T_CALLED("set_field_status(%p,%d)"), (void *)field, status));

  Normalize_Field(field);

  if (status)
    SetStatus(field, _CHANGED);
  else
    ClrStatus(field, _CHANGED);

  RETURN(E_OK);
}

/*---------------------------------------------------------------------------
|   Facility      :  libnform  
|   Function      :  bool field_status(const FIELD *field)
|   
|   Description   :  Retrieve the value of the 'changed' indication flag
|                    for that fields primary buffer. 
|
|   Return Values :  TRUE  - buffer has been changed
|                    FALSE - buffer has not been changed
+--------------------------------------------------------------------------*/
NCURSES_EXPORT(bool)
field_status(const FIELD *field)
{
  T((T_CALLED("field_status(%p)"), (const void *)field));

  returnBool((Normalize_Field(field)->status & _CHANGED) ? TRUE : FALSE);
}

/* fld_stat.c ends here */
