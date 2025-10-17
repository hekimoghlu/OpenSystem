/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 12, 2024.
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

MODULE_ID("$Id: fld_page.c,v 1.12 2012/06/10 00:12:47 tom Exp $")

/*---------------------------------------------------------------------------
|   Facility      :  libnform  
|   Function      :  int set_new_page(FIELD *field, bool new_page_flag)
|   
|   Description   :  Marks the field as the beginning of a new page of 
|                    the form.
|
|   Return Values :  E_OK         - success
|                    E_CONNECTED  - field is connected
+--------------------------------------------------------------------------*/
NCURSES_EXPORT(int)
set_new_page(FIELD *field, bool new_page_flag)
{
  T((T_CALLED("set_new_page(%p,%d)"), (void *)field, new_page_flag));

  Normalize_Field(field);
  if (field->form)
    RETURN(E_CONNECTED);

  if (new_page_flag)
    SetStatus(field, _NEWPAGE);
  else
    ClrStatus(field, _NEWPAGE);

  RETURN(E_OK);
}

/*---------------------------------------------------------------------------
|   Facility      :  libnform  
|   Function      :  bool new_page(const FIELD *field)
|   
|   Description   :  Retrieve the info whether or not the field starts a
|                    new page on the form.
|
|   Return Values :  TRUE  - field starts a new page
|                    FALSE - field doesn't start a new page
+--------------------------------------------------------------------------*/
NCURSES_EXPORT(bool)
new_page(const FIELD *field)
{
  T((T_CALLED("new_page(%p)"), (const void *)field));

  returnBool((Normalize_Field(field)->status & _NEWPAGE) ? TRUE : FALSE);
}

/* fld_page.c ends here */
