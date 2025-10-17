/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 1, 2023.
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
#include "uucp.h"

#if USE_RCS_ID
const char quotes_rcsid[] = "$Id: quotes.c,v 1.2 2002/03/05 19:10:42 ian Rel $";
#endif

#include <ctype.h>

#include "uudefs.h"

/* Copy a string, adding quotes if necessary.  */

char *
zquote_cmd_string (zorig, fbackslashonly)
     const char *zorig;
     boolean fbackslashonly;
{
  const char *z;
  char *zret;
  char *zto;

  if (zorig == NULL)
    return NULL;

  zret = zbufalc (strlen (zorig) * 4 + 1);
  zto = zret;
  for (z = zorig; *z != '\0'; ++z)
    {
      if (*z == '\\')
	{
	  *zto++ = '\\';
	  *zto++ = '\\';
	}
      else if (fbackslashonly || isgraph (BUCHAR (*z)))
	*zto++ = *z;
      else
	{
	  sprintf (zto, "\\%03o", (unsigned int) BUCHAR (*z));
	  zto += strlen (zto);
	}
    }

  *zto = '\0';

  return zret;
}
