/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 26, 2022.
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

MODULE_ID("$Id: fld_arg.c,v 1.13 2012/06/10 00:27:49 tom Exp $")

/*---------------------------------------------------------------------------
|   Facility      :  libnform
|   Function      :  int set_fieldtype_arg(
|                            FIELDTYPE *typ,
|                            void * (* const make_arg)(va_list *),
|                            void * (* const copy_arg)(const void *),
|                            void   (* const free_arg)(void *) )
|
|   Description   :  Connects to the type additional arguments necessary
|                    for a set_field_type call. The various function pointer
|                    arguments are:
|                       make_arg : allocates a structure for the field
|                                  specific parameters.
|                       copy_arg : duplicate the structure created by
|                                  make_arg
|                       free_arg : Release the memory allocated by make_arg
|                                  or copy_arg
|
|                    At least make_arg must be non-NULL.
|                    You may pass NULL for copy_arg and free_arg if your
|                    make_arg function doesn't allocate memory and your
|                    arg fits into the storage for a (void*).
|
|   Return Values :  E_OK           - success
|                    E_BAD_ARGUMENT - invalid argument
+--------------------------------------------------------------------------*/
NCURSES_EXPORT(int)
set_fieldtype_arg(FIELDTYPE *typ,
		  void *(*const make_arg)(va_list *),
		  void *(*const copy_arg)(const void *),
		  void (*const free_arg) (void *))
{
  T((T_CALLED("set_fieldtype_arg(%p,%p,%p,%p)"),
     (void *)typ, make_arg, copy_arg, free_arg));

  if (typ != 0 && make_arg != (void *)0)
    {
      SetStatus(typ, _HAS_ARGS);
      typ->makearg = make_arg;
      typ->copyarg = copy_arg;
      typ->freearg = free_arg;
      RETURN(E_OK);
    }
  RETURN(E_BAD_ARGUMENT);
}

/*---------------------------------------------------------------------------
|   Facility      :  libnform
|   Function      :  void *field_arg(const FIELD *field)
|
|   Description   :  Retrieve pointer to the fields argument structure.
|
|   Return Values :  Pointer to structure or NULL if none is defined.
+--------------------------------------------------------------------------*/
NCURSES_EXPORT(void *)
field_arg(const FIELD *field)
{
  T((T_CALLED("field_arg(%p)"), (const void *)field));
  returnVoidPtr(Normalize_Field(field)->arg);
}

/* fld_arg.c ends here */
