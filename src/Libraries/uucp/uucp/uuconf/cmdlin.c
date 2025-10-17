/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 8, 2024.
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
const char _uuconf_cmdlin_rcsid[] = "$Id: cmdlin.c,v 1.7 2002/03/05 19:10:42 ian Rel $";
#endif

#include <errno.h>
#include <ctype.h>

/* Parse a command line into fields and process it via a command
   table.  The command table functions may keep the memory allocated
   for the line, but they may not keep the memory allocated for the
   argv list.  This function strips # comments.  */

#define CSTACK (16)

int
uuconf_cmd_line (pglobal, zline, qtab, pinfo, pfiunknown, iflags, pblock)
     pointer pglobal;
     char *zline;
     const struct uuconf_cmdtab *qtab;
     pointer pinfo;
     int (*pfiunknown) P((pointer, int, char **, pointer, pointer));
     int iflags;
     pointer pblock;
{
  struct sglobal *qglobal = (struct sglobal *) pglobal;
  char *z;
  int cargs;
  char *azargs[CSTACK];
  char **pzargs;
  int iret;

  if ((iflags & UUCONF_CMDTABFLAG_NOCOMMENTS) == 0)
    {
      /* Any # not preceeded by a backslash starts a comment.  */
      z = zline;
      while ((z = strchr (z, '#')) != NULL)
	{
	  if (z == zline || *(z - 1) != '\\')
	    {
	      *z = '\0';
	      break;
	    }
	  /* Remove the backslash.  */
	  while ((*(z - 1) = *z) != '\0')
	    ++z;
	}
    }

  /* Parse the first CSTACK arguments by hand to avoid malloc.  */

  z = zline;
  cargs = 0;
  pzargs = azargs;
  while (TRUE)
    {
      while (*z != '\0' && isspace (BUCHAR (*z)))
	++z;

      if (*z == '\0')
	break;

      if (cargs >= CSTACK)
	{
	  char **pzsplit;
	  size_t csplit;
	  int cmore;

	  pzsplit = NULL;
	  csplit = 0;
	  cmore = _uuconf_istrsplit (z, '\0', &pzsplit, &csplit);
	  if (cmore < 0)
	    {
	      qglobal->ierrno = errno;
	      return UUCONF_MALLOC_FAILED | UUCONF_ERROR_ERRNO;
	    }

	  pzargs = (char **) malloc ((cmore + CSTACK) * sizeof (char *));
	  if (pzargs == NULL)
	    {
	      qglobal->ierrno = errno;
	      free ((pointer) pzsplit);
	      return UUCONF_MALLOC_FAILED | UUCONF_ERROR_ERRNO;
	    }

	  memcpy ((pointer) pzargs, (pointer) azargs,
		  CSTACK * sizeof (char *));
	  memcpy ((pointer) (pzargs + CSTACK), (pointer) pzsplit,
		  cmore * sizeof (char *));
	  cargs = cmore + CSTACK;

	  free ((pointer) pzsplit);

	  break;
	}

      azargs[cargs] = z;
      ++cargs;

      while (*z != '\0' && ! isspace (BUCHAR (*z)))
	z++;

      if (*z == '\0')
	break;

      *z++ = '\0';
    }

  if (cargs <= 0)
    return UUCONF_CMDTABRET_CONTINUE;

  iret = uuconf_cmd_args (pglobal, cargs, pzargs, qtab, pinfo, pfiunknown,
			  iflags, pblock);

  if (pzargs != azargs)
    free ((pointer) pzargs);

  return iret;
}
