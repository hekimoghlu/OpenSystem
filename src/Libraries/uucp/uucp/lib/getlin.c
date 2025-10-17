/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 10, 2024.
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

/* Read a line from a file, returning the number of characters read.
   This should really return ssize_t.  Returns -1 on error.  */

#define CGETLINE_DEFAULT (63)

int
getline (pzline, pcline, e)
     char **pzline;
     size_t *pcline;
     FILE *e;
{
  char *zput, *zend;
  int bchar;

  if (*pzline == NULL)
    {
      *pzline = (char *) malloc (CGETLINE_DEFAULT);
      if (*pzline == NULL)
	return -1;
      *pcline = CGETLINE_DEFAULT;
    }

  zput = *pzline;
  zend = *pzline + *pcline - 1;

  while ((bchar = getc (e)) != EOF)
    {
      if (zput >= zend)
	{
	  size_t cnew;
	  char *znew;

	  cnew = *pcline * 2 + 1;
	  znew = (char *) realloc ((pointer) *pzline, cnew);
	  if (znew == NULL)
	    return -1;
	  zput = znew + *pcline - 1;
	  zend = znew + cnew - 1;
	  *pzline = znew;
	  *pcline = cnew;
	}

      *zput++ = bchar;

      if (bchar == '\n')
	break;
    }

  if (zput == *pzline)
    return -1;

  *zput = '\0';
  return zput - *pzline;
}
