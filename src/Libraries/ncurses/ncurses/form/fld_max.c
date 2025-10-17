/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 17, 2024.
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

MODULE_ID("$Id: fld_max.c,v 1.13 2013/08/24 22:59:28 tom Exp $")

/*---------------------------------------------------------------------------
|   Facility      :  libnform  
|   Function      :  int set_max_field(FIELD *field, int maxgrow)
|   
|   Description   :  Set the maximum growth for a dynamic field. If maxgrow=0
|                    the field may grow to any possible size.
|
|   Return Values :  E_OK           - success
|                    E_BAD_ARGUMENT - invalid argument
+--------------------------------------------------------------------------*/
NCURSES_EXPORT(int)
set_max_field(FIELD *field, int maxgrow)
{
  T((T_CALLED("set_max_field(%p,%d)"), (void *)field, maxgrow));

  if (!field || (maxgrow < 0))
    RETURN(E_BAD_ARGUMENT);
  else
    {
      bool single_line_field = Single_Line_Field(field);

      if (maxgrow > 0)
	{
	  if ((single_line_field && (maxgrow < field->dcols)) ||
	      (!single_line_field && (maxgrow < field->drows)))
	    RETURN(E_BAD_ARGUMENT);
	}
      field->maxgrow = maxgrow;
      ClrStatus(field, _MAY_GROW);
      if (!((unsigned)field->opts & O_STATIC))
	{
	  if ((maxgrow == 0) ||
	      (single_line_field && (field->dcols < maxgrow)) ||
	      (!single_line_field && (field->drows < maxgrow)))
	    SetStatus(field, _MAY_GROW);
	}
    }
  RETURN(E_OK);
}

/* fld_max.c ends here */
