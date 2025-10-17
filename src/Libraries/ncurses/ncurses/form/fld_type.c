/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 8, 2024.
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

MODULE_ID("$Id: fld_type.c,v 1.16 2010/01/23 21:14:36 tom Exp $")

/*---------------------------------------------------------------------------
|   Facility      :  libnform  
|   Function      :  int set_field_type(FIELD *field, FIELDTYPE *type,...)
|   
|   Description   :  Associate the specified fieldtype with the field.
|                    Certain field types take additional arguments. Look
|                    at the spec of the field types !
|
|   Return Values :  E_OK           - success
|                    E_SYSTEM_ERROR - system error
+--------------------------------------------------------------------------*/
NCURSES_EXPORT(int)
set_field_type(FIELD *field, FIELDTYPE *type,...)
{
  va_list ap;
  int res = E_SYSTEM_ERROR;
  int err = 0;

  T((T_CALLED("set_field_type(%p,%p)"), (void *)field, (void *)type));

  va_start(ap, type);

  Normalize_Field(field);
  _nc_Free_Type(field);

  field->type = type;
  field->arg = (void *)_nc_Make_Argument(field->type, &ap, &err);

  if (err)
    {
      _nc_Free_Argument(field->type, (TypeArgument *)(field->arg));
      field->type = (FIELDTYPE *)0;
      field->arg = (void *)0;
    }
  else
    {
      res = E_OK;
      if (field->type)
	field->type->ref++;
    }

  va_end(ap);
  RETURN(res);
}

/*---------------------------------------------------------------------------
|   Facility      :  libnform  
|   Function      :  FIELDTYPE *field_type(const FIELD *field)
|   
|   Description   :  Retrieve the associated fieldtype for this field.
|
|   Return Values :  Pointer to fieldtype of NULL if none is defined.
+--------------------------------------------------------------------------*/
NCURSES_EXPORT(FIELDTYPE *)
field_type(const FIELD *field)
{
  T((T_CALLED("field_type(%p)"), (const void *)field));
  returnFieldType(Normalize_Field(field)->type);
}

/* fld_type.c ends here */
