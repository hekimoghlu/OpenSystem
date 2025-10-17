/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 3, 2022.
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
 *   Author:  Juergen Pfeifer, 1995-1997,2009                               *
 ****************************************************************************/

#include "form.priv.h"

MODULE_ID("$Id: frm_sub.c,v 1.12 2010/01/23 21:14:36 tom Exp $")

/*---------------------------------------------------------------------------
|   Facility      :  libnform  
|   Function      :  int set_form_sub(FORM *form, WINDOW *win)
|   
|   Description   :  Set the subwindow of the form to win. 
|
|   Return Values :  E_OK       - success
|                    E_POSTED   - form is posted
+--------------------------------------------------------------------------*/
NCURSES_EXPORT(int)
set_form_sub(FORM *form, WINDOW *win)
{
  T((T_CALLED("set_form_sub(%p,%p)"), (void *)form, (void *)win));

  if (form && (form->status & _POSTED))
    RETURN(E_POSTED);
  else
    {
#if NCURSES_SP_FUNCS
      FORM *f = Normalize_Form(form);

      f->sub = win ? win : StdScreen(Get_Form_Screen(f));
      RETURN(E_OK);
#else
      Normalize_Form(form)->sub = win;
      RETURN(E_OK);
#endif
    }
}

/*---------------------------------------------------------------------------
|   Facility      :  libnform  
|   Function      :  WINDOW *form_sub(const FORM *)
|   
|   Description   :  Retrieve the window of the form.
|
|   Return Values :  The pointer to the Subwindow.
+--------------------------------------------------------------------------*/
NCURSES_EXPORT(WINDOW *)
form_sub(const FORM *form)
{
  const FORM *f;

  T((T_CALLED("form_sub(%p)"), (const void *)form));

  f = Normalize_Form(form);
  returnWin(Get_Form_Window(f));
}

/* frm_sub.c ends here */
