/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 9, 2022.
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

MODULE_ID("$Id: fld_attr.c,v 1.11 2010/01/23 21:12:08 tom Exp $")

/*----------------------------------------------------------------------------
  Field-Attribute manipulation routines
  --------------------------------------------------------------------------*/
/* "Template" macro to generate a function to set a fields attribute */
#define GEN_FIELD_ATTR_SET_FCT( name ) \
NCURSES_IMPEXP int NCURSES_API set_field_ ## name (FIELD * field, chtype attr)\
{\
   int res = E_BAD_ARGUMENT;\
   T((T_CALLED("set_field_" #name "(%p,%s)"), field, _traceattr(attr)));\
   if ( attr==A_NORMAL || ((attr & A_ATTRIBUTES)==attr) )\
     {\
       Normalize_Field( field );\
       if (field != 0) \
	 { \
	 if ((field -> name) != attr)\
	   {\
	     field -> name = attr;\
	     res = _nc_Synchronize_Attributes( field );\
	   }\
	 else\
	   {\
	     res = E_OK;\
	   }\
	 }\
     }\
   RETURN(res);\
}

/* "Template" macro to generate a function to get a fields attribute */
#define GEN_FIELD_ATTR_GET_FCT( name ) \
NCURSES_IMPEXP chtype NCURSES_API field_ ## name (const FIELD * field)\
{\
   T((T_CALLED("field_" #name "(%p)"), (const void *) field));\
   returnAttr( A_ATTRIBUTES & (Normalize_Field( field ) -> name) );\
}

/*---------------------------------------------------------------------------
|   Facility      :  libnform
|   Function      :  int set_field_fore(FIELD *field, chtype attr)
|
|   Description   :  Sets the foreground of the field used to display the
|                    field contents.
|
|   Return Values :  E_OK             - success
|                    E_BAD_ARGUMENT   - invalid attributes
|                    E_SYSTEM_ERROR   - system error
+--------------------------------------------------------------------------*/
GEN_FIELD_ATTR_SET_FCT(fore)

/*---------------------------------------------------------------------------
|   Facility      :  libnform
|   Function      :  chtype field_fore(const FIELD *)
|
|   Description   :  Retrieve fields foreground attribute
|
|   Return Values :  The foreground attribute
+--------------------------------------------------------------------------*/
GEN_FIELD_ATTR_GET_FCT(fore)

/*---------------------------------------------------------------------------
|   Facility      :  libnform
|   Function      :  int set_field_back(FIELD *field, chtype attr)
|
|   Description   :  Sets the background of the field used to display the
|                    fields extend.
|
|   Return Values :  E_OK             - success
|                    E_BAD_ARGUMENT   - invalid attributes
|                    E_SYSTEM_ERROR   - system error
+--------------------------------------------------------------------------*/
GEN_FIELD_ATTR_SET_FCT(back)

/*---------------------------------------------------------------------------
|   Facility      :  libnform
|   Function      :  chtype field_back(const
|
|   Description   :  Retrieve fields background attribute
|
|   Return Values :  The background attribute
+--------------------------------------------------------------------------*/
GEN_FIELD_ATTR_GET_FCT(back)

/* fld_attr.c ends here */
