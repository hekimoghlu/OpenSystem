/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 28, 2025.
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
const char _uuconf_ugtlin_rcsid[] = "$Id: ugtlin.c,v 1.9 2002/03/05 19:10:43 ian Rel $";
#endif

/* Read a line from a file with backslash continuations.  This updates
   the qglobal->ilineno count for each additional line it reads.  */

int
_uuconf_getline (qglobal, pzline, pcline, e)
     struct sglobal *qglobal;
     char **pzline;
     size_t *pcline;
     FILE *e;
{
  int ctot;
  char *zline;
  size_t cline;

  ctot = -1;

  zline = NULL;
  cline = 0;

  while (TRUE)
    {
      int cchars;

      if (ctot < 0)
	cchars = getline (pzline, pcline, e);
      else
	cchars = getline (&zline, &cline, e);
      if (cchars < 0)
	{
	  if (zline != NULL)
	    free ((pointer) zline);
	  if (ctot >= 0)
	    return ctot;
	  else
	    return cchars;
	}

      if (ctot < 0)
	ctot = cchars;
      else
	{
	  if (*pcline <= (size_t) (ctot + cchars))
	    {
	      char *znew;

	      if (*pcline > 0)
		znew = (char *) realloc ((pointer) *pzline,
					 (size_t) (ctot + cchars + 1));
	      else
		znew = (char *) malloc ((size_t) (ctot + cchars + 1));
	      if (znew == NULL)
		{
		  free ((pointer) zline);
		  return -1;
		}
	      *pzline = znew;
	      *pcline = ctot + cchars + 1;
	    }

	  memcpy ((pointer) ((*pzline) + ctot), (pointer) zline,
		  (size_t) (cchars + 1));
	  ctot += cchars;
	}

      if (ctot < 2
	  || (*pzline)[ctot - 1] != '\n'
	  || (*pzline)[ctot - 2] != '\\')
	{
	  if (zline != NULL)
	    free ((pointer) zline);
	  return ctot;
	}

      ++qglobal->ilineno;

      ctot -= 2;
      (*pzline)[ctot] = '\0';
    }
}
