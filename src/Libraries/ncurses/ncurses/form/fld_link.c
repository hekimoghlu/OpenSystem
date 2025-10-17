/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 6, 2022.
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

MODULE_ID("$Id: fld_link.c,v 1.13 2012/03/11 00:37:16 tom Exp $")

/*---------------------------------------------------------------------------
|   Facility      :  libnform  
|   Function      :  FIELD *link_field(FIELD *field, int frow, int fcol)  
|   
|   Description   :  Duplicates the field at the specified position. The
|                    new field shares its buffers with the original one,
|                    the attributes are independent.
|                    If an error occurs, errno is set to
|                    
|                    E_BAD_ARGUMENT - invalid argument
|                    E_SYSTEM_ERROR - system error
|
|   Return Values :  Pointer to the new field or NULL if failure
+--------------------------------------------------------------------------*/
NCURSES_EXPORT(FIELD *)
link_field(FIELD *field, int frow, int fcol)
{
  FIELD *New_Field = (FIELD *)0;
  int err = E_BAD_ARGUMENT;

  T((T_CALLED("link_field(%p,%d,%d)"), (void *)field, frow, fcol));
  if (field && (frow >= 0) && (fcol >= 0) &&
      ((err = E_SYSTEM_ERROR) != 0) &&	/* trick: this resets the default error */
      (New_Field = typeMalloc(FIELD, 1)))
    {
      T((T_CREATE("field %p"), (void *)New_Field));
      *New_Field = *_nc_Default_Field;
      New_Field->frow = (short) frow;
      New_Field->fcol = (short) fcol;

      New_Field->link = field->link;
      field->link = New_Field;

      New_Field->buf = field->buf;
      New_Field->rows = field->rows;
      New_Field->cols = field->cols;
      New_Field->nrow = field->nrow;
      New_Field->nbuf = field->nbuf;
      New_Field->drows = field->drows;
      New_Field->dcols = field->dcols;
      New_Field->maxgrow = field->maxgrow;
      New_Field->just = field->just;
      New_Field->fore = field->fore;
      New_Field->back = field->back;
      New_Field->pad = field->pad;
      New_Field->opts = field->opts;
      New_Field->usrptr = field->usrptr;

      if (_nc_Copy_Type(New_Field, field))
	returnField(New_Field);
    }

  if (New_Field)
    free_field(New_Field);

  SET_ERROR(err);
  returnField((FIELD *)0);
}

/* fld_link.c ends here */
