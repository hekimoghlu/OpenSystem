/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 3, 2024.
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
const char _uuconf_dnams_rcsid[] = "$Id: dnams.c,v 1.6 2002/03/05 19:10:42 ian Rel $";
#endif

/* Get all known dialer names.  */

int
uuconf_dialer_names (pglobal, ppzdialers)
     pointer pglobal;
     char ***ppzdialers;
{
  struct sglobal *qglobal = (struct sglobal *) pglobal;
  char **pztaylor;
  char **pzhdb;
  int iret;

  *ppzdialers = NULL;
  pztaylor = NULL;
  pzhdb = NULL;

#if HAVE_TAYLOR_CONFIG
  iret = uuconf_taylor_dialer_names (pglobal, &pztaylor);
  if (iret != UUCONF_SUCCESS)
    return iret;
#endif

#if HAVE_HDB_CONFIG
  if (qglobal->qprocess->fhdb)
    {
      iret = uuconf_hdb_dialer_names (pglobal, &pzhdb);
      if (iret != UUCONF_SUCCESS)
	return iret;
    }
#endif

  if (pzhdb == NULL)
    *ppzdialers = pztaylor;
  else if (pztaylor == NULL)
    *ppzdialers = pzhdb;
  else
    {
      char **pz;

      iret = UUCONF_SUCCESS;

      for (pz = pztaylor; *pz != NULL; pz++)
	{
	  iret = _uuconf_iadd_string (qglobal, *pz, FALSE, TRUE,
				      ppzdialers, (pointer) NULL);
	  if (iret != UUCONF_SUCCESS)
	    break;
	}

      if (iret == UUCONF_SUCCESS)
	{
	  for (pz = pzhdb; *pz != NULL; pz++)
	    {
	      iret = _uuconf_iadd_string (qglobal, *pz, FALSE, TRUE,
					  ppzdialers, (pointer) NULL);
	      if (iret != UUCONF_SUCCESS)
		break;
	    }
	}

      if (pztaylor != NULL)
	free ((pointer) pztaylor);
      if (pzhdb != NULL)
	free ((pointer) pzhdb);
    }

  if (iret == UUCONF_SUCCESS && *ppzdialers == NULL)
    iret = _uuconf_iadd_string (qglobal, (char *) NULL, FALSE, FALSE,
				ppzdialers, (pointer) NULL);

  return iret;
}
