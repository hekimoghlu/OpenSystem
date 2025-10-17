/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 17, 2022.
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

MODULE_ID("$Id: fld_user.c,v 1.16 2010/01/23 21:14:36 tom Exp $")

/*---------------------------------------------------------------------------
|   Facility      :  libnform  
|   Function      :  int set_field_userptr(FIELD *field, void *usrptr)
|   
|   Description   :  Set the pointer that is reserved in any field to store
|                    application relevant informations
|
|   Return Values :  E_OK         - on success
+--------------------------------------------------------------------------*/
NCURSES_EXPORT(int)
set_field_userptr(FIELD *field, void *usrptr)
{
  T((T_CALLED("set_field_userptr(%p,%p)"), (void *)field, (void *)usrptr));

  Normalize_Field(field)->usrptr = usrptr;
  RETURN(E_OK);
}

/*---------------------------------------------------------------------------
|   Facility      :  libnform  
|   Function      :  void *field_userptr(const FIELD *field)
|   
|   Description   :  Return the pointer that is reserved in any field to
|                    store application relevant informations.
|
|   Return Values :  Value of pointer. If no such pointer has been set,
|                    NULL is returned
+--------------------------------------------------------------------------*/
NCURSES_EXPORT(void *)
field_userptr(const FIELD *field)
{
  T((T_CALLED("field_userptr(%p)"), (const void *)field));
  returnVoidPtr(Normalize_Field(field)->usrptr);
}

/* fld_user.c ends here */
