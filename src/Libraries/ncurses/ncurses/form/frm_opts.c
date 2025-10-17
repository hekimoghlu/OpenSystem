/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 28, 2023.
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

MODULE_ID("$Id: frm_opts.c,v 1.17 2013/08/24 22:58:47 tom Exp $")

/*---------------------------------------------------------------------------
|   Facility      :  libnform  
|   Function      :  int set_form_opts(FORM *form, Form_Options opts)
|   
|   Description   :  Turns on the named options and turns off all the
|                    remaining options for that form.
|
|   Return Values :  E_OK              - success
|                    E_BAD_ARGUMENT    - invalid options
+--------------------------------------------------------------------------*/
NCURSES_EXPORT(int)
set_form_opts(FORM *form, Form_Options opts)
{
  T((T_CALLED("set_form_opts(%p,%d)"), (void *)form, opts));

  opts &= (Form_Options) ALL_FORM_OPTS;
  if ((unsigned)opts & ~ALL_FORM_OPTS)
    RETURN(E_BAD_ARGUMENT);
  else
    {
      Normalize_Form(form)->opts = opts;
      RETURN(E_OK);
    }
}

/*---------------------------------------------------------------------------
|   Facility      :  libnform  
|   Function      :  Form_Options form_opts(const FORM *)
|   
|   Description   :  Retrieves the current form options.
|
|   Return Values :  The option flags.
+--------------------------------------------------------------------------*/
NCURSES_EXPORT(Form_Options)
form_opts(const FORM *form)
{
  T((T_CALLED("form_opts(%p)"), (const void *)form));
  returnCode((Form_Options) ((unsigned)Normalize_Form(form)->opts & ALL_FORM_OPTS));
}

/*---------------------------------------------------------------------------
|   Facility      :  libnform  
|   Function      :  int form_opts_on(FORM *form, Form_Options opts)
|   
|   Description   :  Turns on the named options; no other options are 
|                    changed.
|
|   Return Values :  E_OK            - success 
|                    E_BAD_ARGUMENT  - invalid options
+--------------------------------------------------------------------------*/
NCURSES_EXPORT(int)
form_opts_on(FORM *form, Form_Options opts)
{
  T((T_CALLED("form_opts_on(%p,%d)"), (void *)form, opts));

  opts &= (Form_Options) ALL_FORM_OPTS;
  if ((unsigned)opts & ~ALL_FORM_OPTS)
    RETURN(E_BAD_ARGUMENT);
  else
    {
      Normalize_Form(form)->opts |= opts;
      RETURN(E_OK);
    }
}

/*---------------------------------------------------------------------------
|   Facility      :  libnform  
|   Function      :  int form_opts_off(FORM *form, Form_Options opts)
|   
|   Description   :  Turns off the named options; no other options are 
|                    changed.
|
|   Return Values :  E_OK            - success 
|                    E_BAD_ARGUMENT  - invalid options
+--------------------------------------------------------------------------*/
NCURSES_EXPORT(int)
form_opts_off(FORM *form, Form_Options opts)
{
  T((T_CALLED("form_opts_off(%p,%d)"), (void *)form, opts));

  opts &= (Form_Options) ALL_FORM_OPTS;
  if ((unsigned)opts & ~ALL_FORM_OPTS)
    RETURN(E_BAD_ARGUMENT);
  else
    {
      Normalize_Form(form)->opts &= ~opts;
      RETURN(E_OK);
    }
}

/* frm_opts.c ends here */
