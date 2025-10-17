/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 16, 2024.
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
const char _uuconf_tlocnm_rcsid[] = "$Id: tlocnm.c,v 1.7 2002/03/05 19:10:43 ian Rel $";
#endif

#include <errno.h>

/* Get the local name to use, based on the login name, from the Taylor
   UUCP configuration files.  This could probably be done in a
   slightly more intelligent fashion, but no matter what it has to
   read the systems files.  */

int
uuconf_taylor_login_localname (pglobal, zlogin, pzname)
     pointer pglobal;
     const char *zlogin;
     char **pzname;
{
  struct sglobal *qglobal = (struct sglobal *) pglobal;
  char **pznames, **pz;
  int iret;

  if (! qglobal->qprocess->fread_syslocs)
    {
      iret = _uuconf_iread_locations (qglobal);
      if (iret != UUCONF_SUCCESS)
	return iret;
    }

  /* As a simple optimization, if there is no "myname" command we can
     simply return immediately.  */
  if (! qglobal->qprocess->fuses_myname)
    {
      *pzname = NULL;
      return UUCONF_NOT_FOUND;
    }

  iret = uuconf_taylor_system_names (pglobal, &pznames, 0);
  if (iret != UUCONF_SUCCESS)
    return iret;

  *pzname = NULL;
  iret = UUCONF_NOT_FOUND;

  for (pz = pznames; *pz != NULL; pz++)
    {
      struct uuconf_system ssys;
      struct uuconf_system *qsys;

      iret = uuconf_system_info (pglobal, *pz, &ssys);
      if (iret != UUCONF_SUCCESS)
	break;

      for (qsys = &ssys; qsys != NULL; qsys = qsys->uuconf_qalternate)
	{
	  if (qsys->uuconf_zlocalname != NULL
	      && qsys->uuconf_fcalled
	      && qsys->uuconf_zcalled_login != NULL
	      && strcmp (qsys->uuconf_zcalled_login, zlogin) == 0)
	    {
	      *pzname = strdup (qsys->uuconf_zlocalname);
	      if (*pzname != NULL)
		iret = UUCONF_SUCCESS;
	      else
		{
		  qglobal->ierrno = errno;
		  iret = UUCONF_MALLOC_FAILED | UUCONF_ERROR_ERRNO;
		}
	      break;
	    }
	}

      (void) uuconf_system_free (pglobal, &ssys);

      if (qsys != NULL)
	break;

      iret = UUCONF_NOT_FOUND;
    }

  for (pz = pznames; *pz != NULL; pz++)
    free ((pointer) *pz);
  free ((pointer) pznames);

  return iret;
}
