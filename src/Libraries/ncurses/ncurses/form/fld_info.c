/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 8, 2025.
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

MODULE_ID("$Id: fld_info.c,v 1.11 2010/01/23 21:14:35 tom Exp $")

/*---------------------------------------------------------------------------
|   Facility      :  libnform  
|   Function      :  int field_info(const FIELD *field,
|                                   int *rows, int *cols,
|                                   int *frow, int *fcol,
|                                   int *nrow, int *nbuf)
|   
|   Description   :  Retrieve infos about the fields creation parameters.
|
|   Return Values :  E_OK           - success
|                    E_BAD_ARGUMENT - invalid field pointer
+--------------------------------------------------------------------------*/
NCURSES_EXPORT(int)
field_info(const FIELD *field,
	   int *rows, int *cols,
	   int *frow, int *fcol,
	   int *nrow, int *nbuf)
{
  T((T_CALLED("field_info(%p,%p,%p,%p,%p,%p,%p)"),
     (const void *)field,
     (void *)rows, (void *)cols,
     (void *)frow, (void *)fcol,
     (void *)nrow, (void *)nbuf));

  if (!field)
    RETURN(E_BAD_ARGUMENT);

  if (rows)
    *rows = field->rows;
  if (cols)
    *cols = field->cols;
  if (frow)
    *frow = field->frow;
  if (fcol)
    *fcol = field->fcol;
  if (nrow)
    *nrow = field->nrow;
  if (nbuf)
    *nbuf = field->nbuf;
  RETURN(E_OK);
}

/*---------------------------------------------------------------------------
|   Facility      :  libnform  
|   Function      :  int dynamic_field_info(const FIELD *field,
|                                           int *drows, int *dcols,
|                                           int *maxgrow)
|   
|   Description   :  Retrieve informations about a dynamic fields current
|                    dynamic parameters.
|
|   Return Values :  E_OK           - success
|                    E_BAD_ARGUMENT - invalid argument
+--------------------------------------------------------------------------*/
NCURSES_EXPORT(int)
dynamic_field_info(const FIELD *field, int *drows, int *dcols, int *maxgrow)
{
  T((T_CALLED("dynamic_field_info(%p,%p,%p,%p)"),
     (const void *)field,
     (void *)drows,
     (void *)dcols,
     (void *)maxgrow));

  if (!field)
    RETURN(E_BAD_ARGUMENT);

  if (drows)
    *drows = field->drows;
  if (dcols)
    *dcols = field->dcols;
  if (maxgrow)
    *maxgrow = field->maxgrow;

  RETURN(E_OK);
}

/* fld_info.c ends here */
