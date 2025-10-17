/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 31, 2022.
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
const char _uuconf_cmdfil_rcsid[] = "$Id: cmdfil.c,v 1.7 2002/03/05 19:10:42 ian Rel $";
#endif

#include <errno.h>

/* Read and parse commands from a file, updating uuconf_lineno as
   appropriate.  */

int
uuconf_cmd_file (pglobal, e, qtab, pinfo, pfiunknown, iflags, pblock)
     pointer pglobal;
     FILE *e;
     const struct uuconf_cmdtab *qtab;
     pointer pinfo;
     int (*pfiunknown) P((pointer, int, char **, pointer, pointer));
     int iflags;
     pointer pblock;
{
  struct sglobal *qglobal = (struct sglobal *) pglobal;
  boolean fcont;
  char *zline;
  size_t cline;
  int iret;

  fcont = (iflags & UUCONF_CMDTABFLAG_BACKSLASH) != 0;

  zline = NULL;
  cline = 0;

  iret = UUCONF_SUCCESS;

  qglobal->ilineno = 0;

  while ((fcont
	  ? _uuconf_getline (qglobal, &zline, &cline, e)
	  : getline (&zline, &cline, e)) > 0)
    {
      ++qglobal->ilineno;

      iret = uuconf_cmd_line (pglobal, zline, qtab, pinfo, pfiunknown,
			      iflags, pblock);

      if ((iret & UUCONF_CMDTABRET_KEEP) != 0)
	{
	  iret &=~ UUCONF_CMDTABRET_KEEP;

	  if (pblock != NULL)
	    {
	      if (uuconf_add_block (pblock, zline) != 0)
		{
		  qglobal->ierrno = errno;
		  iret = (UUCONF_MALLOC_FAILED
			  | UUCONF_ERROR_ERRNO
			  | UUCONF_ERROR_LINENO);
		  break;
		}
	    }

	  zline = NULL;
	  cline = 0;
	}

      if ((iret & UUCONF_CMDTABRET_EXIT) != 0)
	{
	  iret &=~ UUCONF_CMDTABRET_EXIT;
	  if (iret != UUCONF_SUCCESS)
	    iret |= UUCONF_ERROR_LINENO;
	  break;
	}

      iret = UUCONF_SUCCESS;
    }

  if (zline != NULL)
    free ((pointer) zline);

  return iret;
}
