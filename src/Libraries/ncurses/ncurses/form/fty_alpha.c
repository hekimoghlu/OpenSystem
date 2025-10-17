/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 7, 2024.
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
/***************************************************************************
*                                                                          *
*  Author : Juergen Pfeifer                                                *
*                                                                          *
***************************************************************************/

#include "form.priv.h"

MODULE_ID("$Id: fty_alpha.c,v 1.26 2010/01/23 21:14:36 tom Exp $")

#define thisARG alphaARG

typedef struct
  {
    int width;
  }
thisARG;

/*---------------------------------------------------------------------------
|   Facility      :  libnform
|   Function      :  static void *Generic_This_Type(va_list *ap)
|
|   Description   :  Allocate structure for alpha type argument.
|
|   Return Values :  Pointer to argument structure or NULL on error
+--------------------------------------------------------------------------*/
static void *
Generic_This_Type(void *arg)
{
  thisARG *argp = (thisARG *) 0;

  if (arg)
    {
      argp = typeMalloc(thisARG, 1);

      if (argp)
	{
	  T((T_CREATE("thisARG %p"), (void *)argp));
	  argp->width = *((int *)arg);
	}
    }
  return ((void *)argp);
}

/*---------------------------------------------------------------------------
|   Facility      :  libnform
|   Function      :  static void *Make_This_Type(va_list *ap)
|
|   Description   :  Allocate structure for alpha type argument.
|
|   Return Values :  Pointer to argument structure or NULL on error
+--------------------------------------------------------------------------*/
static void *
Make_This_Type(va_list *ap)
{
  int w = va_arg(*ap, int);

  return Generic_This_Type((void *)&w);
}

/*---------------------------------------------------------------------------
|   Facility      :  libnform
|   Function      :  static void *Copy_This_Type(const void * argp)
|
|   Description   :  Copy structure for alpha type argument.
|
|   Return Values :  Pointer to argument structure or NULL on error.
+--------------------------------------------------------------------------*/
static void *
Copy_This_Type(const void *argp)
{
  const thisARG *ap = (const thisARG *)argp;
  thisARG *result = typeMalloc(thisARG, 1);

  if (result)
    {
      T((T_CREATE("thisARG %p"), (void *)result));
      *result = *ap;
    }

  return ((void *)result);
}

/*---------------------------------------------------------------------------
|   Facility      :  libnform
|   Function      :  static void Free_This_Type(void *argp)
|
|   Description   :  Free structure for alpha type argument.
|
|   Return Values :  -
+--------------------------------------------------------------------------*/
static void
Free_This_Type(void *argp)
{
  if (argp)
    free(argp);
}

/*---------------------------------------------------------------------------
|   Facility      :  libnform
|   Function      :  static bool Check_This_Character(
|                                      int c,
|                                      const void *argp)
|
|   Description   :  Check a character for the alpha type.
|
|   Return Values :  TRUE  - character is valid
|                    FALSE - character is invalid
+--------------------------------------------------------------------------*/
static bool
Check_This_Character(int c, const void *argp GCC_UNUSED)
{
#if USE_WIDEC_SUPPORT
  if (iswalpha((wint_t) c))
    return TRUE;
#endif
  return (isalpha(UChar(c)) ? TRUE : FALSE);
}

/*---------------------------------------------------------------------------
|   Facility      :  libnform
|   Function      :  static bool Check_This_Field(
|                                      FIELD *field,
|                                      const void *argp)
|
|   Description   :  Validate buffer content to be a valid alpha value
|
|   Return Values :  TRUE  - field is valid
|                    FALSE - field is invalid
+--------------------------------------------------------------------------*/
static bool
Check_This_Field(FIELD *field, const void *argp)
{
  int width = ((const thisARG *)argp)->width;
  unsigned char *bp = (unsigned char *)field_buffer(field, 0);
  bool result = (width < 0);

  Check_CTYPE_Field(result, bp, width, Check_This_Character);
  return (result);
}

static FIELDTYPE typeTHIS =
{
  _HAS_ARGS | _RESIDENT,
  1,				/* this is mutable, so we can't be const */
  (FIELDTYPE *)0,
  (FIELDTYPE *)0,
  Make_This_Type,
  Copy_This_Type,
  Free_This_Type,
  INIT_FT_FUNC(Check_This_Field),
  INIT_FT_FUNC(Check_This_Character),
  INIT_FT_FUNC(NULL),
  INIT_FT_FUNC(NULL),
#if NCURSES_INTEROP_FUNCS
  Generic_This_Type
#endif
};

NCURSES_EXPORT_VAR(FIELDTYPE*) TYPE_ALPHA = &typeTHIS;

#if NCURSES_INTEROP_FUNCS
/* The next routines are to simplify the use of ncurses from
   programming languages with restictions on interop with C level
   constructs (e.g. variable access or va_list + ellipsis constructs)
*/
NCURSES_EXPORT(FIELDTYPE *)
_nc_TYPE_ALPHA(void)
{
  return TYPE_ALPHA;
}
#endif

/* fty_alpha.c ends here */
