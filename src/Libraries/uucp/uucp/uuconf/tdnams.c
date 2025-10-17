/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 6, 2023.
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
const char _uuconf_tdnams_rcsid[] = "$Id: tdnams.c,v 1.9 2002/03/05 19:10:43 ian Rel $";
#endif

#include <errno.h>

static int indialer P((pointer pglobal, int argc, char **argv, pointer pvar,
		       pointer pinfo));

/* Get the names of all the dialers from the Taylor UUCP configuration
   files.  */

int
uuconf_taylor_dialer_names (pglobal, ppzdialers)
     pointer pglobal;
     char ***ppzdialers;
{
  struct sglobal *qglobal = (struct sglobal *) pglobal;
  struct uuconf_cmdtab as[2];
  char **pz;
  int iret;
  
  *ppzdialers = NULL;

  as[0].uuconf_zcmd = "dialer";
  as[0].uuconf_itype = UUCONF_CMDTABTYPE_FN | 2;
  as[0].uuconf_pvar = (pointer) ppzdialers;
  as[0].uuconf_pifn = indialer;

  as[1].uuconf_zcmd = NULL;

  iret = UUCONF_SUCCESS;

  for (pz = qglobal->qprocess->pzdialfiles; *pz != NULL; pz++)
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
			      (uuconf_cmdtabfn) NULL,
			      UUCONF_CMDTABFLAG_BACKSLASH,
			      (pointer) NULL);

      (void) fclose (e);

      if (iret != UUCONF_SUCCESS)
	break;
    }

  if (iret != UUCONF_SUCCESS)
    {
      qglobal->zfilename = *pz;
      return iret | UUCONF_ERROR_FILENAME;
    }

  if (*ppzdialers == NULL)
    iret = _uuconf_iadd_string (qglobal, (char *) NULL, FALSE, FALSE,
				ppzdialers, (pointer) NULL);

  return UUCONF_SUCCESS;
}

/* Add a dialer name to the list.  */

/*ARGSUSED*/
static int
indialer (pglobal, argc, argv, pvar, pinfo)
     pointer pglobal;
     int argc ATTRIBUTE_UNUSED;
     char **argv;
     pointer pvar;
     pointer pinfo ATTRIBUTE_UNUSED;
{
  struct sglobal *qglobal = (struct sglobal *) pglobal;
  char ***ppzdialers = (char ***) pvar;
  int iret;

  iret = _uuconf_iadd_string (qglobal, argv[1], TRUE, TRUE, ppzdialers,
			      (pointer) NULL);
  if (iret != UUCONF_SUCCESS)
    iret |= UUCONF_CMDTABRET_EXIT;
  return iret;
}
