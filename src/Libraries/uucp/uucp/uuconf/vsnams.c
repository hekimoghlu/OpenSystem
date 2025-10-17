/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 2, 2023.
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
const char _uuconf_vsnams_rcsid[] = "$Id: vsnams.c,v 1.11 2002/03/05 19:10:43 ian Rel $";
#endif

#include <errno.h>

/* Get all the system names from the V2 L.sys file.  This code does
   not support aliases, although some V2 versions do have an L-aliases
   file.  */

/*ARGSUSED*/
int
uuconf_v2_system_names (pglobal, ppzsystems, falias)
     pointer pglobal;
     char ***ppzsystems;
     int falias ATTRIBUTE_UNUSED;
{
  struct sglobal *qglobal = (struct sglobal *) pglobal;
  FILE *e;
  int iret;
  char *zline;
  size_t cline;

  *ppzsystems = NULL;

  e = fopen (qglobal->qprocess->zv2systems, "r");
  if (e == NULL)
    {
      if (FNO_SUCH_FILE ())
	return _uuconf_iadd_string (qglobal, (char *) NULL, FALSE, FALSE,
				    ppzsystems, (pointer) NULL);
      qglobal->ierrno = errno;
      qglobal->zfilename = qglobal->qprocess->zv2systems;
      return (UUCONF_FOPEN_FAILED
	      | UUCONF_ERROR_ERRNO
	      | UUCONF_ERROR_FILENAME);
    }

  qglobal->ilineno = 0;
  iret = UUCONF_SUCCESS;

  zline = NULL;
  cline = 0;
  while (_uuconf_getline (qglobal, &zline, &cline, e) > 0)
    {
      char *zname;

      ++qglobal->ilineno;

      /* Skip leading whitespace to get to the system name.  Then cut
	 the system name off at the first whitespace, comment, or
	 newline.  */
      zname = zline + strspn (zline, " \t");
      zname[strcspn (zname, " \t#\n")] = '\0';
      if (*zname == '\0')
	continue;

      iret = _uuconf_iadd_string (qglobal, zname, TRUE, TRUE, ppzsystems,
				  (pointer) NULL);
      if (iret != UUCONF_SUCCESS)
	break;
    }

  (void) fclose (e);
  if (zline != NULL)
    free ((pointer) zline);

  if (iret != UUCONF_SUCCESS)
    {
      qglobal->zfilename = qglobal->qprocess->zv2systems;
      return iret | UUCONF_ERROR_FILENAME | UUCONF_ERROR_LINENO;
    }

  if (*ppzsystems == NULL)
    iret = _uuconf_iadd_string (qglobal, (char *) NULL, FALSE, FALSE,
				ppzsystems, (pointer) NULL);

  return iret;
}
