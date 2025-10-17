/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 23, 2023.
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

MODULE_ID("$Id: fld_newftyp.c,v 1.19 2010/01/23 21:14:36 tom Exp $")

static FIELDTYPE default_fieldtype =
{
  0,				/* status                                      */
  0L,				/* reference count                             */
  (FIELDTYPE *)0,		/* pointer to left  operand                    */
  (FIELDTYPE *)0,		/* pointer to right operand                    */
  NULL,				/* makearg function                            */
  NULL,				/* copyarg function                            */
  NULL,				/* freearg function                            */
  INIT_FT_FUNC(NULL),		/* field validation function                   */
  INIT_FT_FUNC(NULL),		/* Character check function                    */
  INIT_FT_FUNC(NULL),		/* enumerate next function                     */
  INIT_FT_FUNC(NULL),		/* enumerate previous function                 */
#if NCURSES_INTEROP_FUNCS
  NULL				/* generic callback alternative to makearg     */
#endif
};

NCURSES_EXPORT_VAR(FIELDTYPE *)
_nc_Default_FieldType = &default_fieldtype;

/*---------------------------------------------------------------------------
|   Facility      :  libnform
|   Function      :  FIELDTYPE *new_fieldtype(
|                       bool (* const field_check)(FIELD *,const void *),
|                       bool (* const char_check) (int, const void *) )
|
|   Description   :  Create a new fieldtype. The application programmer must
|                    write a field_check and a char_check function and give
|                    them as input to this call.
|                    If an error occurs, errno is set to
|                       E_BAD_ARGUMENT  - invalid arguments
|                       E_SYSTEM_ERROR  - system error (no memory)
|
|   Return Values :  Fieldtype pointer or NULL if error occurred
+--------------------------------------------------------------------------*/
NCURSES_EXPORT(FIELDTYPE *)
new_fieldtype(bool (*const field_check) (FIELD *, const void *),
	      bool (*const char_check) (int, const void *))
{
  FIELDTYPE *nftyp = (FIELDTYPE *)0;

  T((T_CALLED("new_fieldtype(%p,%p)"), field_check, char_check));
  if ((field_check) || (char_check))
    {
      nftyp = typeMalloc(FIELDTYPE, 1);

      if (nftyp)
	{
	  T((T_CREATE("fieldtype %p"), (void *)nftyp));
	  *nftyp = default_fieldtype;
#if NCURSES_INTEROP_FUNCS
	  nftyp->fieldcheck.ofcheck = field_check;
	  nftyp->charcheck.occheck = char_check;
#else
	  nftyp->fcheck = field_check;
	  nftyp->ccheck = char_check;
#endif
	}
      else
	{
	  SET_ERROR(E_SYSTEM_ERROR);
	}
    }
  else
    {
      SET_ERROR(E_BAD_ARGUMENT);
    }
  returnFieldType(nftyp);
}

/*---------------------------------------------------------------------------
|   Facility      :  libnform
|   Function      :  int free_fieldtype(FIELDTYPE *typ)
|
|   Description   :  Release the memory associated with this fieldtype.
|
|   Return Values :  E_OK            - success
|                    E_CONNECTED     - there are fields referencing the type
|                    E_BAD_ARGUMENT  - invalid fieldtype pointer
+--------------------------------------------------------------------------*/
NCURSES_EXPORT(int)
free_fieldtype(FIELDTYPE *typ)
{
  T((T_CALLED("free_fieldtype(%p)"), (void *)typ));

  if (!typ)
    RETURN(E_BAD_ARGUMENT);

  if (typ->ref != 0)
    RETURN(E_CONNECTED);

  if (typ->status & _RESIDENT)
    RETURN(E_CONNECTED);

  if (typ->status & _LINKED_TYPE)
    {
      if (typ->left)
	typ->left->ref--;
      if (typ->right)
	typ->right->ref--;
    }
  free(typ);
  RETURN(E_OK);
}

/* fld_newftyp.c ends here */
