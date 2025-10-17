/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 26, 2025.
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
const char _uuconf_sinfo_rcsid[] = "$Id: sinfo.c,v 1.6 2002/03/05 19:10:42 ian Rel $";
#endif

/* Get information about a particular system.  We combine the
   definitions for this system from each type of configuration file,
   by passing what we have so far into each one.  */

int
uuconf_system_info (pglobal, zsystem, qsys)
     pointer pglobal;
     const char *zsystem;
     struct uuconf_system *qsys;
{
  struct sglobal *qglobal = (struct sglobal *) pglobal;
  int iret;
  boolean fgot;

  fgot = FALSE;

#if HAVE_TAYLOR_CONFIG
  iret = _uuconf_itaylor_system_internal (qglobal, zsystem, qsys);
  if (iret == UUCONF_SUCCESS)
    fgot = TRUE;
  else if (iret != UUCONF_NOT_FOUND)
    return iret;
#endif

#if HAVE_V2_CONFIG
  if (qglobal->qprocess->fv2)
    {
      struct uuconf_system *q;
      struct uuconf_system sv2;

      if (fgot)
	q = &sv2;
      else
	q = qsys;
      iret = _uuconf_iv2_system_internal (qglobal, zsystem, q);
      if (iret == UUCONF_SUCCESS)
	{
	  if (fgot)
	    {
	      iret = _uuconf_isystem_default (qglobal, qsys, &sv2, TRUE);
	      if (iret != UUCONF_SUCCESS)
		return iret;
	    }
	  fgot = TRUE;
	}
      else if (iret != UUCONF_NOT_FOUND)
	return iret;
    }
#endif

#if HAVE_HDB_CONFIG
  if (qglobal->qprocess->fhdb)
    {
      struct uuconf_system *q;
      struct uuconf_system shdb;

      if (fgot)
	q = &shdb;
      else
	q = qsys;
      iret = _uuconf_ihdb_system_internal (qglobal, zsystem, q);
      if (iret == UUCONF_SUCCESS)
	{
	  if (fgot)
	    {
	      iret = _uuconf_isystem_default (qglobal, qsys, &shdb, TRUE);
	      if (iret != UUCONF_SUCCESS)
		return iret;
	    }
	  fgot = TRUE;
	}
      else if (iret != UUCONF_NOT_FOUND)
	return iret;
    }
#endif

  if (! fgot)
    return UUCONF_NOT_FOUND;

  return _uuconf_isystem_basic_default (qglobal, qsys);
}
