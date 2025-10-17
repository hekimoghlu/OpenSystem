/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 29, 2022.
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
 *   Author:  Thomas E. Dickey                                              *
 ****************************************************************************/

#include "form.priv.h"

MODULE_ID("$Id: f_trace.c,v 1.2 2010/01/23 21:14:36 tom Exp $")

NCURSES_EXPORT(FIELD **)
_nc_retrace_field_ptr(FIELD **code)
{
  T((T_RETURN("%p"), (void *)code));
  return code;
}

NCURSES_EXPORT(FIELD *)
_nc_retrace_field(FIELD *code)
{
  T((T_RETURN("%p"), (void *)code));
  return code;
}

NCURSES_EXPORT(FIELDTYPE *)
_nc_retrace_field_type(FIELDTYPE *code)
{
  T((T_RETURN("%p"), (void *)code));
  return code;
}

NCURSES_EXPORT(FORM *)
_nc_retrace_form(FORM *code)
{
  T((T_RETURN("%p"), (void *)code));
  return code;
}

NCURSES_EXPORT(Form_Hook)
_nc_retrace_form_hook(Form_Hook code)
{
  T((T_RETURN("%p"), code));
  return code;
}
