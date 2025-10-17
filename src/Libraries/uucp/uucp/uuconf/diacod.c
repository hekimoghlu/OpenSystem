/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 15, 2022.
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
const char _uuconf_diacod_rcsid[] = "$Id: diacod.c,v 1.12 2002/03/05 19:10:42 ian Rel $";
#endif

#include <errno.h>

static int idcode P((pointer pglobal, int argc, char **argv,
		     pointer pinfo, pointer pvar));

/* Get the name of the UUCP log file.  */

int
uuconf_dialcode (pglobal, zdial, pznum)
     pointer pglobal;
     const char *zdial;
     char **pznum;
{
  struct sglobal *qglobal = (struct sglobal *) pglobal;
  struct uuconf_cmdtab as[2];
  char **pz;
  int iret;

  as[0].uuconf_zcmd = zdial;
  as[0].uuconf_itype = UUCONF_CMDTABTYPE_FN | 0;
  as[0].uuconf_pvar = (pointer) pznum;
  as[0].uuconf_pifn = idcode;

  as[1].uuconf_zcmd = NULL;

  *pznum = NULL;

  iret = UUCONF_SUCCESS;

  for (pz = qglobal->qprocess->pzdialcodefiles; *pz != NULL; pz++)
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

      iret = uuconf_cmd_file (pglobal, e, as, (pointer) NULL,
			      (uuconf_cmdtabfn) NULL, 0, (pointer) NULL);
      (void) fclose (e);

      if (iret != UUCONF_SUCCESS || *pznum != NULL)
	break;
    }

  if (iret != UUCONF_SUCCESS)
    {
      qglobal->zfilename = *pz;
      iret |= UUCONF_ERROR_FILENAME;
    }
  else if (*pznum == NULL)
    iret = UUCONF_NOT_FOUND;

  return iret;
}

/* This is called if the dialcode is found.  It copies the number into
   the heap and gets out of reading the file.  */

/*ARGSUSED*/
static int
idcode (pglobal, argc, argv, pvar, pinfo)
     pointer pglobal;
     int argc;
     char **argv;
     pointer pvar;
     pointer pinfo ATTRIBUTE_UNUSED;
{
  struct sglobal *qglobal = (struct sglobal *) pglobal;
  char **pznum = (char **) pvar;

  if (argc == 1)
    {
      *pznum = malloc (1);
      if (*pznum != NULL)
	**pznum = '\0';
    }
  else if (argc == 2)
    *pznum = strdup (argv[1]);
  else
    return UUCONF_SYNTAX_ERROR | UUCONF_CMDTABRET_EXIT;

  if (*pznum == NULL)
    {
      qglobal->ierrno = errno;
      return (UUCONF_MALLOC_FAILED
	      | UUCONF_ERROR_ERRNO
	      | UUCONF_CMDTABRET_EXIT);
    }

  return UUCONF_CMDTABRET_EXIT;
}
