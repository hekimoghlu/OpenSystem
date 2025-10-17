/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 13, 2022.
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

MODULE_ID("$Id: frm_scale.c,v 1.10 2010/01/23 21:14:36 tom Exp $")

/*---------------------------------------------------------------------------
|   Facility      :  libnform  
|   Function      :  int scale_form( const FORM *form, int *rows, int *cols )
|   
|   Description   :  Retrieve size of form
|
|   Return Values :  E_OK              - no error
|                    E_BAD_ARGUMENT    - invalid form pointer
|                    E_NOT_CONNECTED   - no fields connected to form
+--------------------------------------------------------------------------*/
NCURSES_EXPORT(int)
scale_form(const FORM *form, int *rows, int *cols)
{
  T((T_CALLED("scale_form(%p,%p,%p)"),
     (const void *)form,
     (void *)rows,
     (void *)cols));

  if (!form)
    RETURN(E_BAD_ARGUMENT);

  if (!(form->field))
    RETURN(E_NOT_CONNECTED);

  if (rows)
    *rows = form->rows;
  if (cols)
    *cols = form->cols;

  RETURN(E_OK);
}

/* frm_scale.c ends here */
