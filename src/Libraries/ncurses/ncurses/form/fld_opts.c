/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 13, 2025.
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

MODULE_ID("$Id: fld_opts.c,v 1.12 2010/01/23 21:14:36 tom Exp $")

/*----------------------------------------------------------------------------
  Field-Options manipulation routines
  --------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------
|   Facility      :  libnform  
|   Function      :  int set_field_opts(FIELD *field, Field_Options opts)
|   
|   Description   :  Turns on the named options for this field and turns
|                    off all the remaining options.
|
|   Return Values :  E_OK            - success
|                    E_CURRENT       - the field is the current field
|                    E_BAD_ARGUMENT  - invalid options
|                    E_SYSTEM_ERROR  - system error
+--------------------------------------------------------------------------*/
NCURSES_EXPORT(int)
set_field_opts(FIELD *field, Field_Options opts)
{
  int res = E_BAD_ARGUMENT;

  T((T_CALLED("set_field_opts(%p,%d)"), (void *)field, opts));

  opts &= ALL_FIELD_OPTS;
  if (!(opts & ~ALL_FIELD_OPTS))
    res = _nc_Synchronize_Options(Normalize_Field(field), opts);
  RETURN(res);
}

/*---------------------------------------------------------------------------
|   Facility      :  libnform  
|   Function      :  Field_Options field_opts(const FIELD *field)
|   
|   Description   :  Retrieve the fields options.
|
|   Return Values :  The options.
+--------------------------------------------------------------------------*/
NCURSES_EXPORT(Field_Options)
field_opts(const FIELD *field)
{
  T((T_CALLED("field_opts(%p)"), (const void *)field));

  returnCode(ALL_FIELD_OPTS & Normalize_Field(field)->opts);
}

/*---------------------------------------------------------------------------
|   Facility      :  libnform  
|   Function      :  int field_opts_on(FIELD *field, Field_Options opts)
|   
|   Description   :  Turns on the named options for this field and all the 
|                    remaining options are unchanged.
|
|   Return Values :  E_OK            - success
|                    E_CURRENT       - the field is the current field
|                    E_BAD_ARGUMENT  - invalid options
|                    E_SYSTEM_ERROR  - system error
+--------------------------------------------------------------------------*/
NCURSES_EXPORT(int)
field_opts_on(FIELD *field, Field_Options opts)
{
  int res = E_BAD_ARGUMENT;

  T((T_CALLED("field_opts_on(%p,%d)"), (void *)field, opts));

  opts &= ALL_FIELD_OPTS;
  if (!(opts & ~ALL_FIELD_OPTS))
    {
      Normalize_Field(field);
      res = _nc_Synchronize_Options(field, field->opts | opts);
    }
  RETURN(res);
}

/*---------------------------------------------------------------------------
|   Facility      :  libnform  
|   Function      :  int field_opts_off(FIELD *field, Field_Options opts)
|   
|   Description   :  Turns off the named options for this field and all the 
|                    remaining options are unchanged.
|
|   Return Values :  E_OK            - success
|                    E_CURRENT       - the field is the current field
|                    E_BAD_ARGUMENT  - invalid options
|                    E_SYSTEM_ERROR  - system error
+--------------------------------------------------------------------------*/
NCURSES_EXPORT(int)
field_opts_off(FIELD *field, Field_Options opts)
{
  int res = E_BAD_ARGUMENT;

  T((T_CALLED("field_opts_off(%p,%d)"), (void *)field, opts));

  opts &= ALL_FIELD_OPTS;
  if (!(opts & ~ALL_FIELD_OPTS))
    {
      Normalize_Field(field);
      res = _nc_Synchronize_Options(field, field->opts & ~opts);
    }
  RETURN(res);
}

/* fld_opts.c ends here */
