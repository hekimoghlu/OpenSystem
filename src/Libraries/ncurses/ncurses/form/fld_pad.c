/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 11, 2023.
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

MODULE_ID("$Id: fld_pad.c,v 1.10 2010/01/23 21:14:36 tom Exp $")

/*---------------------------------------------------------------------------
|   Facility      :  libnform  
|   Function      :  int set_field_pad(FIELD *field, int ch)
|   
|   Description   :  Set the pad character used to fill the field. This must
|                    be a printable character.
|
|   Return Values :  E_OK           - success
|                    E_BAD_ARGUMENT - invalid field pointer or pad character
|                    E_SYSTEM_ERROR - system error
+--------------------------------------------------------------------------*/
NCURSES_EXPORT(int)
set_field_pad(FIELD *field, int ch)
{
  int res = E_BAD_ARGUMENT;

  T((T_CALLED("set_field_pad(%p,%d)"), (void *)field, ch));

  Normalize_Field(field);
  if (isprint(UChar(ch)))
    {
      if (field->pad != ch)
	{
	  field->pad = ch;
	  res = _nc_Synchronize_Attributes(field);
	}
      else
	res = E_OK;
    }
  RETURN(res);
}

/*---------------------------------------------------------------------------
|   Facility      :  libnform  
|   Function      :  int field_pad(const FIELD *field)
|   
|   Description   :  Retrieve the fields pad character.
|
|   Return Values :  The pad character.
+--------------------------------------------------------------------------*/
NCURSES_EXPORT(int)
field_pad(const FIELD *field)
{
  T((T_CALLED("field_pad(%p)"), (const void *)field));

  returnCode(Normalize_Field(field)->pad);
}

/* fld_pad.c ends here */
