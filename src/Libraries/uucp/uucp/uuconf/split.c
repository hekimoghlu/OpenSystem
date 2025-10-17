/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 19, 2023.
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
#include "uucnfi.h"

#if USE_RCS_ID
const char _uuconf_split_rcsid[] = "$Id: split.c,v 1.6 2002/03/05 19:10:42 ian Rel $";
#endif

#include <ctype.h>

/* Split a string into tokens.  The bsep argument is the separator to
   use.  If it is the null byte, white space is used as the separator,
   and leading white space is discarded.  Otherwise, each occurrence
   of the separator character delimits a field (and thus some fields
   may be empty).  The array and size arguments may be used to reuse
   the same memory.  This function is not tied to uuconf; the only way
   it can fail is if malloc or realloc fails.  */

int
_uuconf_istrsplit (zline, bsep, ppzsplit, pcsplit)
     register char *zline;
     int bsep;
     char ***ppzsplit;
     size_t *pcsplit;
{
  size_t i;

  i = 0;

  while (TRUE)
    {
      if (bsep == '\0')
	{
	  while (isspace (BUCHAR (*zline)))
	    ++zline;
	  if (*zline == '\0')
	    break;
	}

      if (i >= *pcsplit)
	{
	  char **pznew;
	  size_t cnew;

	  if (*pcsplit == 0)
	    {
	      cnew = 8;
	      pznew = (char **) malloc (cnew * sizeof (char *));
	    }
	  else
	    {
	      cnew = *pcsplit * 2;
	      pznew = (char **) realloc ((pointer) *ppzsplit,
					 cnew * sizeof (char *));
	    }
	  if (pznew == NULL)
	    return -1;
	  *ppzsplit = pznew;
	  *pcsplit = cnew;
	}

      (*ppzsplit)[i] = zline;
      ++i;

      if (bsep == '\0')
	{
	  while (*zline != '\0' && ! isspace (BUCHAR (*zline)))
	    ++zline;
	}
      else
	{
	  while (*zline != '\0' && *zline != bsep)
	    ++zline;
	}

      if (*zline == '\0')
	break;

      *zline++ = '\0';
    }

  return i;
}
