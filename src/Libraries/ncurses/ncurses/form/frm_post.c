/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 13, 2025.
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

MODULE_ID("$Id: frm_post.c,v 1.11 2012/06/10 00:27:49 tom Exp $")

/*---------------------------------------------------------------------------
|   Facility      :  libnform  
|   Function      :  int post_form(FORM * form)
|   
|   Description   :  Writes the form into its associated subwindow.
|
|   Return Values :  E_OK              - success
|                    E_BAD_ARGUMENT    - invalid form pointer
|                    E_POSTED          - form already posted
|                    E_NOT_CONNECTED   - no fields connected to form
|                    E_NO_ROOM         - form doesn't fit into subwindow
|                    E_SYSTEM_ERROR    - system error
+--------------------------------------------------------------------------*/
NCURSES_EXPORT(int)
post_form(FORM *form)
{
  WINDOW *formwin;
  int err;
  int page;

  T((T_CALLED("post_form(%p)"), (void *)form));

  if (!form)
    RETURN(E_BAD_ARGUMENT);

  if (form->status & _POSTED)
    RETURN(E_POSTED);

  if (!(form->field))
    RETURN(E_NOT_CONNECTED);

  formwin = Get_Form_Window(form);
  if ((form->cols > getmaxx(formwin)) || (form->rows > getmaxy(formwin)))
    RETURN(E_NO_ROOM);

  /* reset form->curpage to an invald value. This forces Set_Form_Page
     to do the page initialization which is required by post_form.
   */
  page = form->curpage;
  form->curpage = -1;
  if ((err = _nc_Set_Form_Page(form, page, form->current)) != E_OK)
    RETURN(err);

  SetStatus(form, _POSTED);

  Call_Hook(form, forminit);
  Call_Hook(form, fieldinit);

  _nc_Refresh_Current_Field(form);
  RETURN(E_OK);
}

/*---------------------------------------------------------------------------
|   Facility      :  libnform  
|   Function      :  int unpost_form(FORM * form)
|   
|   Description   :  Erase form from its associated subwindow.
|
|   Return Values :  E_OK            - success
|                    E_BAD_ARGUMENT  - invalid form pointer
|                    E_NOT_POSTED    - form isn't posted
|                    E_BAD_STATE     - called from a hook routine
+--------------------------------------------------------------------------*/
NCURSES_EXPORT(int)
unpost_form(FORM *form)
{
  T((T_CALLED("unpost_form(%p)"), (void *)form));

  if (!form)
    RETURN(E_BAD_ARGUMENT);

  if (!(form->status & _POSTED))
    RETURN(E_NOT_POSTED);

  if (form->status & _IN_DRIVER)
    RETURN(E_BAD_STATE);

  Call_Hook(form, fieldterm);
  Call_Hook(form, formterm);

  werase(Get_Form_Window(form));
  delwin(form->w);
  form->w = (WINDOW *)0;
  ClrStatus(form, _POSTED);
  RETURN(E_OK);
}

/* frm_post.c ends here */
