/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 20, 2022.
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

MODULE_ID("$Id: fld_ftchoice.c,v 1.13 2012/06/10 00:27:49 tom Exp $")

/*---------------------------------------------------------------------------
|   Facility      :  libnform  
|   Function      :  int set_fieldtype_choice(
|                          FIELDTYPE *typ,
|                          bool (* const next_choice)(FIELD *,const void *),
|                          bool (* const prev_choice)(FIELD *,const void *))
|
|   Description   :  Define implementation of enumeration requests.
|
|   Return Values :  E_OK           - success
|                    E_BAD_ARGUMENT - invalid arguments
+--------------------------------------------------------------------------*/
NCURSES_EXPORT(int)
set_fieldtype_choice(FIELDTYPE *typ,
		     bool (*const next_choice) (FIELD *, const void *),
		     bool (*const prev_choice) (FIELD *, const void *))
{
  T((T_CALLED("set_fieldtype_choice(%p,%p,%p)"), (void *)typ, next_choice, prev_choice));

  if (!typ || !next_choice || !prev_choice)
    RETURN(E_BAD_ARGUMENT);

  SetStatus(typ, _HAS_CHOICE);
#if NCURSES_INTEROP_FUNCS
  typ->enum_next.onext = next_choice;
  typ->enum_prev.oprev = prev_choice;
#else
  typ->next = next_choice;
  typ->prev = prev_choice;
#endif
  RETURN(E_OK);
}

/* fld_ftchoice.c ends here */
