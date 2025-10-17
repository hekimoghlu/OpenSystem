/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 13, 2024.
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
const char _uuconf_snams_rcsid[] = "$Id: snams.c,v 1.6 2002/03/05 19:10:42 ian Rel $";
#endif

/* Get all known system names.  */

int
uuconf_system_names (pglobal, ppzsystems, falias)
     pointer pglobal;
     char ***ppzsystems;
     int falias;
{
  struct sglobal *qglobal = (struct sglobal *) pglobal;
  char **pztaylor;
  char **pzv2;
  char **pzhdb;
  int iret;

  *ppzsystems = NULL;
  pztaylor = NULL;
  pzv2 = NULL;
  pzhdb = NULL;

#if HAVE_TAYLOR_CONFIG
  iret = uuconf_taylor_system_names (pglobal, &pztaylor, falias);
  if (iret != UUCONF_SUCCESS)
    return iret;
#endif

#if HAVE_V2_CONFIG
  if (qglobal->qprocess->fv2)
    {
      iret = uuconf_v2_system_names (pglobal, &pzv2, falias);
      if (iret != UUCONF_SUCCESS)
	return iret;
    }
#endif

#if HAVE_HDB_CONFIG
  if (qglobal->qprocess->fhdb)
    {
      iret = uuconf_hdb_system_names (pglobal, &pzhdb, falias);
      if (iret != UUCONF_SUCCESS)
	return iret;
    }
#endif

  if (pzv2 == NULL && pzhdb == NULL)
    *ppzsystems = pztaylor;
  else if (pztaylor == NULL && pzhdb == NULL)
    *ppzsystems = pzv2;
  else if (pztaylor == NULL && pzv2 == NULL)
    *ppzsystems = pzhdb;
  else
    {
      char **pz;

      iret = UUCONF_SUCCESS;

      if (pztaylor != NULL)
	{
	  for (pz = pztaylor; *pz != NULL; pz++)
	    {
	      iret = _uuconf_iadd_string (qglobal, *pz, FALSE, TRUE,
					  ppzsystems, (pointer) NULL);
	      if (iret != UUCONF_SUCCESS)
		break;
	    }
	}

      if (pzv2 != NULL && iret == UUCONF_SUCCESS)
	{
	  for (pz = pzv2; *pz != NULL; pz++)
	    {
	      iret = _uuconf_iadd_string (qglobal, *pz, FALSE, TRUE,
					  ppzsystems, (pointer) NULL);
	      if (iret != UUCONF_SUCCESS)
		break;
	    }
	}

      if (pzhdb != NULL && iret == UUCONF_SUCCESS)
	{
	  for (pz = pzhdb; *pz != NULL; pz++)
	    {
	      iret = _uuconf_iadd_string (qglobal, *pz, FALSE, TRUE,
					  ppzsystems, (pointer) NULL);
	      if (iret != UUCONF_SUCCESS)
		break;
	    }
	}

      if (pztaylor != NULL)
	free ((pointer) pztaylor);
      if (pzv2 != NULL)
	free ((pointer) pzv2);
      if (pzhdb != NULL)
	free ((pointer) pzhdb);
    }

  if (iret == UUCONF_SUCCESS && *ppzsystems == NULL)
    iret = _uuconf_iadd_string (qglobal, (char *) NULL, FALSE, FALSE,
				ppzsystems, (pointer) NULL);

  return iret;
}
