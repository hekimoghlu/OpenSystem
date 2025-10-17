/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 8, 2021.
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
const char _uuconf_hlocnm_rcsid[] = "$Id: hlocnm.c,v 1.8 2002/03/05 19:10:42 ian Rel $";
#endif

#include <errno.h>

/* Get the local name to use, based on the login name, from the HDB
   configuration files.  */

int
uuconf_hdb_login_localname (pglobal, zlogin, pzname)
     pointer pglobal;
     const char *zlogin;
     char **pzname;
{
  struct sglobal *qglobal = (struct sglobal *) pglobal;
  struct shpermissions *qperm;

  if (! qglobal->qprocess->fhdb_read_permissions)
    {
      int iret;

      iret = _uuconf_ihread_permissions (qglobal);
      if (iret != UUCONF_SUCCESS)
	return iret;
    }

  for (qperm = qglobal->qprocess->qhdb_permissions;
       qperm != NULL;
       qperm = qperm->qnext)
    {
      if (qperm->zmyname != NULL
	  && qperm->zmyname != (char *) &_uuconf_unset
	  && qperm->pzlogname != NULL
	  && qperm->pzlogname != (char **) &_uuconf_unset)
	{
	  char **pz;

	  for (pz = qperm->pzlogname; *pz != NULL; pz++)
	    {
	      if (strcmp (*pz, zlogin) == 0)
		{
		  *pzname = strdup (qperm->zmyname);
		  if (*pzname == NULL)
		    {
		      qglobal->ierrno = errno;
		      return UUCONF_MALLOC_FAILED | UUCONF_ERROR_ERRNO;
		    }
		  return UUCONF_SUCCESS;
		}
	    }
	}
    }

  *pzname = NULL;
  return UUCONF_NOT_FOUND;
}
