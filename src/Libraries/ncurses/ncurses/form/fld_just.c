/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 10, 2022.
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

MODULE_ID("$Id: fld_just.c,v 1.13 2012/03/11 00:37:16 tom Exp $")

/*---------------------------------------------------------------------------
|   Facility      :  libnform  
|   Function      :  int set_field_just(FIELD *field, int just)
|   
|   Description   :  Set the fields type of justification.
|
|   Return Values :  E_OK            - success
|                    E_BAD_ARGUMENT  - one of the arguments was incorrect
|                    E_SYSTEM_ERROR  - system error
+--------------------------------------------------------------------------*/
NCURSES_EXPORT(int)
set_field_just(FIELD *field, int just)
{
  int res = E_BAD_ARGUMENT;

  T((T_CALLED("set_field_just(%p,%d)"), (void *)field, just));

  if ((just == NO_JUSTIFICATION) ||
      (just == JUSTIFY_LEFT) ||
      (just == JUSTIFY_CENTER) ||
      (just == JUSTIFY_RIGHT))
    {
      Normalize_Field(field);
      if (field->just != just)
	{
	  field->just = (short) just;
	  res = _nc_Synchronize_Attributes(field);
	}
      else
	res = E_OK;
    }
  RETURN(res);
}

/*---------------------------------------------------------------------------
|   Facility      :  libnform  
|   Function      :  int field_just( const FIELD *field )
|   
|   Description   :  Retrieve the fields type of justification
|
|   Return Values :  The justification type.
+--------------------------------------------------------------------------*/
NCURSES_EXPORT(int)
field_just(const FIELD *field)
{
  T((T_CALLED("field_just(%p)"), (const void *)field));
  returnCode(Normalize_Field(field)->just);
}

/* fld_just.c ends here */
