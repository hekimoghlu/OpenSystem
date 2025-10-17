/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 18, 2023.
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
const char _uuconf_hdial_rcsid[] = "$Id: hdial.c,v 1.7 2002/03/05 19:10:42 ian Rel $";
#endif

#include <errno.h>
#include <ctype.h>

/* Find a dialer in the HDB configuration files by name.  */

int
uuconf_hdb_dialer_info (pglobal, zname, qdialer)
     pointer pglobal;
     const char *zname;
     struct uuconf_dialer *qdialer;
{
  struct sglobal *qglobal = (struct sglobal *) pglobal;
  char **pz;
  char *zline;
  size_t cline;
  char **pzsplit;
  size_t csplit;
  int iret;

  zline = NULL;
  cline = 0;
  pzsplit = NULL;
  csplit = 0;

  iret = UUCONF_NOT_FOUND;

  for (pz = qglobal->qprocess->pzhdb_dialers; *pz != NULL; pz++)
    {
      FILE *e;
      int cchars;

      qglobal->ilineno = 0;

      e = fopen (*pz, "r");
      if (e == NULL)
	{
	  if (FNO_SUCH_FILE ())
	    continue;
	  qglobal->ierrno = errno;
	  iret = UUCONF_FOPEN_FAILED | UUCONF_ERROR_ERRNO;
	  break;
	}

      while ((cchars = _uuconf_getline (qglobal, &zline, &cline, e)) > 0)
	{
	  int ctoks;
	  pointer pblock;

	  ++qglobal->ilineno;

	  --cchars;
	  if (zline[cchars] == '\n')
	    zline[cchars] = '\0';
	  if (isspace (BUCHAR (zline[0])) || zline[0] == '#')
	    continue;

	  ctoks = _uuconf_istrsplit (zline, '\0', &pzsplit, &csplit);
	  if (ctoks < 0)
	    {
	      qglobal->ierrno = errno;
	      iret = UUCONF_MALLOC_FAILED | UUCONF_ERROR_ERRNO;
	      break;
	    }

	  if (ctoks < 1)
	    continue;

	  if (strcmp (zname, pzsplit[0]) != 0)
	    continue;

	  /* We found the dialer.  */
	  pblock = uuconf_malloc_block ();
	  if (pblock == NULL)
	    {
	      qglobal->ierrno = errno;
	      iret = UUCONF_MALLOC_FAILED | UUCONF_ERROR_ERRNO;
	      break;
	    }
	  if (uuconf_add_block (pblock, zline) != 0)
	    {
	      qglobal->ierrno = errno;
	      uuconf_free_block (pblock);
	      iret = UUCONF_MALLOC_FAILED | UUCONF_ERROR_ERRNO;
	      break;
	    }
	  zline = NULL;

	  _uuconf_uclear_dialer (qdialer);
	  qdialer->uuconf_zname = pzsplit[0];
	  qdialer->uuconf_palloc = pblock;

	  if (ctoks > 1)
	    {
	      /* The second field is characters to send instead of "="
		 and "-" in phone numbers.  */
	      if (strcmp (pzsplit[1], "\"\"") == 0)
		{
		  char *zsubs;
		  char bnext;

		  zsubs = pzsplit[1];
		  bnext = *zsubs;
		  while (bnext != '\0')
		    {
		      if (bnext == '=') 
			qdialer->uuconf_zdialtone = zsubs + 1;
		      else if (bnext == '-')
			qdialer->uuconf_zpause = zsubs + 1;
		      if (zsubs[1] == '\0')
			break;
		      zsubs += 2;
		      bnext = *zsubs;
		      *zsubs = '\0';
		    }
		}

	      /* Any remaining fields form a chat script.  */
	      if (ctoks > 2)
		{
		  pzsplit[1] = (char *) "chat";
		  iret = _uuconf_ichat_cmd (qglobal, ctoks - 1,
					    pzsplit + 1,
					    &qdialer->uuconf_schat,
					    pblock);
		  iret &=~ UUCONF_CMDTABRET_KEEP;
		  if (iret != UUCONF_SUCCESS)
		    {
		      uuconf_free_block (pblock);
		      break;
		    }
		}
	    }

	  iret = UUCONF_SUCCESS;
	  break;
	}

      (void) fclose (e);

      if (iret != UUCONF_NOT_FOUND)
	break;
    }

  if (zline != NULL)
    free ((pointer) zline);
  if (pzsplit != NULL)
    free ((pointer) pzsplit);

  if (iret != UUCONF_SUCCESS && iret != UUCONF_NOT_FOUND)
    {
      qglobal->zfilename = *pz;
      iret |= UUCONF_ERROR_FILENAME | UUCONF_ERROR_LINENO;
    }

  return iret;
}
