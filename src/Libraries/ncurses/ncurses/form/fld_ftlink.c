/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 30, 2024.
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

MODULE_ID("$Id: fld_ftlink.c,v 1.15 2012/06/10 00:27:49 tom Exp $")

/*---------------------------------------------------------------------------
|   Facility      :  libnform  
|   Function      :  FIELDTYPE *link_fieldtype(
|                                FIELDTYPE *type1,
|                                FIELDTYPE *type2)
|   
|   Description   :  Create a new fieldtype built from the two given types.
|                    They are connected by an logical 'OR'.
|                    If an error occurs, errno is set to                    
|                       E_BAD_ARGUMENT  - invalid arguments
|                       E_SYSTEM_ERROR  - system error (no memory)
|
|   Return Values :  Fieldtype pointer or NULL if error occurred.
+--------------------------------------------------------------------------*/
NCURSES_EXPORT(FIELDTYPE *)
link_fieldtype(FIELDTYPE *type1, FIELDTYPE *type2)
{
  FIELDTYPE *nftyp = (FIELDTYPE *)0;

  T((T_CALLED("link_fieldtype(%p,%p)"), (void *)type1, (void *)type2));
  if (type1 && type2)
    {
      nftyp = typeMalloc(FIELDTYPE, 1);

      if (nftyp)
	{
	  T((T_CREATE("fieldtype %p"), (void *)nftyp));
	  *nftyp = *_nc_Default_FieldType;
	  SetStatus(nftyp, _LINKED_TYPE);
	  if ((type1->status & _HAS_ARGS) || (type2->status & _HAS_ARGS))
	    SetStatus(nftyp, _HAS_ARGS);
	  if ((type1->status & _HAS_CHOICE) || (type2->status & _HAS_CHOICE))
	    SetStatus(nftyp, _HAS_CHOICE);
	  nftyp->left = type1;
	  nftyp->right = type2;
	  type1->ref++;
	  type2->ref++;
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

/* fld_ftlink.c ends here */
