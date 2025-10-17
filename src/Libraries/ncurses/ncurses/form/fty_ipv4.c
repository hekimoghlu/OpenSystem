/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 28, 2023.
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
*  Author : Per Foreby, perf@efd.lth.se                                    *
*                                                                          *
***************************************************************************/

#include "form.priv.h"

MODULE_ID("$Id: fty_ipv4.c,v 1.10 2009/11/07 20:17:58 tom Exp $")

/*---------------------------------------------------------------------------
|   Facility      :  libnform  
|   Function      :  static bool Check_IPV4_Field(
|                                      FIELD * field,
|                                      const void * argp)
|   
|   Description   :  Validate buffer content to be a valid IP number (Ver. 4)
|
|   Return Values :  TRUE  - field is valid
|                    FALSE - field is invalid
+--------------------------------------------------------------------------*/
static bool
Check_IPV4_Field(FIELD *field, const void *argp GCC_UNUSED)
{
  char *bp = field_buffer(field, 0);
  int num = 0, len;
  unsigned int d1, d2, d3, d4;

  if (isdigit(UChar(*bp)))	/* Must start with digit */
    {
      num = sscanf(bp, "%u.%u.%u.%u%n", &d1, &d2, &d3, &d4, &len);
      if (num == 4)
	{
	  bp += len;		/* Make bp point to what sscanf() left */
	  while (isspace(UChar(*bp)))
	    bp++;		/* Allow trailing whitespace */
	}
    }
  return ((num != 4 || *bp || d1 > 255 || d2 > 255
	   || d3 > 255 || d4 > 255) ? FALSE : TRUE);
}

/*---------------------------------------------------------------------------
|   Facility      :  libnform  
|   Function      :  static bool Check_IPV4_Character(
|                                      int c, 
|                                      const void *argp )
|   
|   Description   :  Check a character for unsigned type or period.
|
|   Return Values :  TRUE  - character is valid
|                    FALSE - character is invalid
+--------------------------------------------------------------------------*/
static bool
Check_IPV4_Character(int c, const void *argp GCC_UNUSED)
{
  return ((isdigit(UChar(c)) || (c == '.')) ? TRUE : FALSE);
}

static FIELDTYPE typeIPV4 =
{
  _RESIDENT,
  1,				/* this is mutable, so we can't be const */
  (FIELDTYPE *)0,
  (FIELDTYPE *)0,
  NULL,
  NULL,
  NULL,
  INIT_FT_FUNC(Check_IPV4_Field),
  INIT_FT_FUNC(Check_IPV4_Character),
  INIT_FT_FUNC(NULL),
  INIT_FT_FUNC(NULL),
#if NCURSES_INTEROP_FUNCS
  NULL
#endif
};

NCURSES_EXPORT_VAR(FIELDTYPE*) TYPE_IPV4 = &typeIPV4;

#if NCURSES_INTEROP_FUNCS
/* The next routines are to simplify the use of ncurses from
   programming languages with restictions on interop with C level
   constructs (e.g. variable access or va_list + ellipsis constructs)
*/
NCURSES_EXPORT(FIELDTYPE *)
_nc_TYPE_IPV4(void)
{
  return TYPE_IPV4;
}
#endif

/* fty_ipv4.c ends here */
