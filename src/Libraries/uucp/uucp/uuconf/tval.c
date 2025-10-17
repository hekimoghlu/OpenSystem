/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 7, 2025.
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
const char _uuconf_tval_rcsid[] = "$Id: tval.c,v 1.6 2002/03/05 19:10:43 ian Rel $";
#endif

/* Validate a login name for a system using Taylor UUCP configuration
   files.  This assumes that the zcalled_login field is either NULL or
   "ANY".  If makes sure that the login name does not appear in some
   other "called-login" command listing systems not including this
   one.  */

int
uuconf_taylor_validate (pglobal, qsys, zlogin)
     pointer pglobal;
     const struct uuconf_system *qsys;
     const char *zlogin;
{
  struct sglobal *qglobal = (struct sglobal *) pglobal;
  struct svalidate *q;

  if (! qglobal->qprocess->fread_syslocs)
    {
      int iret;

      iret = _uuconf_iread_locations (qglobal);
      if (iret != UUCONF_SUCCESS)
	return iret;
    }

  for (q = qglobal->qprocess->qvalidate; q != NULL; q = q->qnext)
    {
      if (strcmp (q->zlogname, zlogin) == 0)
	{
	  char **pz;

	  for (pz = q->pzmachines; *pz != NULL; pz++)
	    if (strcmp (*pz, qsys->uuconf_zname) == 0)
	      return UUCONF_SUCCESS;

	  return UUCONF_NOT_FOUND;
	}
    }

  return UUCONF_SUCCESS;
}
