/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 5, 2023.
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
const char _uuconf_hsnams_rcsid[] = "$Id: hsnams.c,v 1.7 2002/03/05 19:10:42 ian Rel $";
#endif

#include <errno.h>
#include <ctype.h>

/* Get all the system names from the HDB Systems file.  We have to
   read the Permissions file in order to support aliases.  */

int
uuconf_hdb_system_names (pglobal, ppzsystems, falias)
     pointer pglobal;
     char ***ppzsystems;
     int falias;
{
  struct sglobal *qglobal = (struct sglobal *) pglobal;
  int iret;
  char *zline;
  size_t cline;
  char **pz;

  *ppzsystems = NULL;

  iret = UUCONF_SUCCESS;

  zline = NULL;
  cline = 0;

  for (pz = qglobal->qprocess->pzhdb_systems; *pz != NULL; pz++)
    {
      FILE *e;

      e = fopen (*pz, "r");
      if (e == NULL)
	{
	  if (FNO_SUCH_FILE ())
	    continue;
	  qglobal->ierrno = errno;
	  iret = UUCONF_FOPEN_FAILED | UUCONF_ERROR_ERRNO;
	  break;
	}
      
      qglobal->ilineno = 0;

      while (_uuconf_getline (qglobal, &zline, &cline, e) > 0)
	{
	  ++qglobal->ilineno;

	  /* Lines beginning with whitespace are treated as comments.
	     No system name can contain a '#', which is another
	     comment character, so eliminating the first '#' does no
	     harm and catches comments.  */
	  zline[strcspn (zline, " \t#\n")] = '\0';
	  if (*zline == '\0')
	    continue;

	  iret = _uuconf_iadd_string (qglobal, zline, TRUE, TRUE,
				      ppzsystems, (pointer) NULL);
	  if (iret != UUCONF_SUCCESS)
	    {
	      iret |= UUCONF_ERROR_LINENO;
	      break;
	    }
	}

      (void) fclose (e);
    }

  if (zline != NULL)
    free ((pointer) zline);

  if (iret != UUCONF_SUCCESS)
    {
      qglobal->zfilename = *pz;
      return iret | UUCONF_ERROR_FILENAME;
    }

  /* If we are supposed to return aliases, we must read the
     Permissions file.  */
  if (falias)
    {
      struct shpermissions *q;

      if (! qglobal->qprocess->fhdb_read_permissions)
	{
	  iret = _uuconf_ihread_permissions (qglobal);
	  if (iret != UUCONF_SUCCESS)
	    return iret;
	}

      for (q = qglobal->qprocess->qhdb_permissions;
	   q != NULL;
	   q = q->qnext)
	{
	  pz = q->pzalias;
	  if (pz == NULL || pz == (char **) &_uuconf_unset)
	    continue;

	  for (; *pz != NULL; pz++)
	    {
	      iret = _uuconf_iadd_string (qglobal, *pz, TRUE, TRUE,
					  ppzsystems, (pointer) NULL);
	      if (iret != UUCONF_SUCCESS)
		return iret;
	    }
	}
    }

  if (*ppzsystems == NULL)
    iret = _uuconf_iadd_string (qglobal, (char *) NULL, FALSE, FALSE,
				ppzsystems, (pointer) NULL);

  return iret;
}
