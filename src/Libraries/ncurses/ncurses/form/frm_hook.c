/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 16, 2022.
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

MODULE_ID("$Id: frm_hook.c,v 1.16 2012/03/11 00:37:16 tom Exp $")

/* "Template" macro to generate function to set application specific hook */
#define GEN_HOOK_SET_FUNCTION( typ, name ) \
NCURSES_IMPEXP int NCURSES_API set_ ## typ ## _ ## name (FORM *form, Form_Hook func)\
{\
   T((T_CALLED("set_" #typ"_"#name"(%p,%p)"), (void *) form, func));\
   (Normalize_Form( form ) -> typ ## name) = func ;\
   RETURN(E_OK);\
}

/* "Template" macro to generate function to get application specific hook */
#define GEN_HOOK_GET_FUNCTION( typ, name ) \
NCURSES_IMPEXP Form_Hook NCURSES_API typ ## _ ## name ( const FORM *form )\
{\
   T((T_CALLED(#typ "_" #name "(%p)"), (const void *) form));\
   returnFormHook( Normalize_Form( form ) -> typ ## name );\
}

/*---------------------------------------------------------------------------
|   Facility      :  libnform
|   Function      :  int set_field_init(FORM *form, Form_Hook f)
|
|   Description   :  Assigns an application defined initialization function
|                    to be called when the form is posted and just after
|                    the current field changes.
|
|   Return Values :  E_OK      - success
+--------------------------------------------------------------------------*/
GEN_HOOK_SET_FUNCTION(field, init)

/*---------------------------------------------------------------------------
|   Facility      :  libnform
|   Function      :  Form_Hook field_init(const FORM *form)
|
|   Description   :  Retrieve field initialization routine address.
|
|   Return Values :  The address or NULL if no hook defined.
+--------------------------------------------------------------------------*/
GEN_HOOK_GET_FUNCTION(field, init)

/*---------------------------------------------------------------------------
|   Facility      :  libnform
|   Function      :  int set_field_term(FORM *form, Form_Hook f)
|
|   Description   :  Assigns an application defined finalization function
|                    to be called when the form is unposted and just before
|                    the current field changes.
|
|   Return Values :  E_OK      - success
+--------------------------------------------------------------------------*/
GEN_HOOK_SET_FUNCTION(field, term)

/*---------------------------------------------------------------------------
|   Facility      :  libnform
|   Function      :  Form_Hook field_term(const FORM *form)
|
|   Description   :  Retrieve field finalization routine address.
|
|   Return Values :  The address or NULL if no hook defined.
+--------------------------------------------------------------------------*/
GEN_HOOK_GET_FUNCTION(field, term)

/*---------------------------------------------------------------------------
|   Facility      :  libnform
|   Function      :  int set_form_init(FORM *form, Form_Hook f)
|
|   Description   :  Assigns an application defined initialization function
|                    to be called when the form is posted and just after
|                    a page change.
|
|   Return Values :  E_OK       - success
+--------------------------------------------------------------------------*/
GEN_HOOK_SET_FUNCTION(form, init)

/*---------------------------------------------------------------------------
|   Facility      :  libnform
|   Function      :  Form_Hook form_init(const FORM *form)
|
|   Description   :  Retrieve form initialization routine address.
|
|   Return Values :  The address or NULL if no hook defined.
+--------------------------------------------------------------------------*/
GEN_HOOK_GET_FUNCTION(form, init)

/*---------------------------------------------------------------------------
|   Facility      :  libnform
|   Function      :  int set_form_term(FORM *form, Form_Hook f)
|
|   Description   :  Assigns an application defined finalization function
|                    to be called when the form is unposted and just before
|                    a page change.
|
|   Return Values :  E_OK       - success
+--------------------------------------------------------------------------*/
GEN_HOOK_SET_FUNCTION(form, term)

/*---------------------------------------------------------------------------
|   Facility      :  libnform
|   Function      :  Form_Hook form_term(const FORM *form)
|
|   Description   :  Retrieve form finalization routine address.
|
|   Return Values :  The address or NULL if no hook defined.
+--------------------------------------------------------------------------*/
GEN_HOOK_GET_FUNCTION(form, term)

/* frm_hook.c ends here */
