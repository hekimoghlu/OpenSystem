/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 1, 2022.
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
const char _uuconf_init_rcsid[] = "$Id: init.c,v 1.8 2002/03/05 19:10:42 ian Rel $";
#endif

/* Initialize the UUCP configuration file reading routines.  This is
   just a generic routine which calls the type specific routines.  */

/*ARGSUSED*/
int
uuconf_init (ppglobal, zprogram, zname)
     pointer *ppglobal;
     const char *zprogram;
     const char *zname;
{
  struct sglobal **pqglob = (struct sglobal **) ppglobal;
  int iret;

  iret = UUCONF_NOT_FOUND;

  *pqglob = NULL;

#if HAVE_TAYLOR_CONFIG
  iret = uuconf_taylor_init (ppglobal, zprogram, zname);
  if (iret != UUCONF_SUCCESS)
    return iret;
#endif

#if HAVE_V2_CONFIG
  if (*pqglob == NULL || (*pqglob)->qprocess->fv2)
    {
      iret = uuconf_v2_init (ppglobal);
      if (iret != UUCONF_SUCCESS)
	return iret;
    }
#endif

#if HAVE_HDB_CONFIG
  if (*pqglob == NULL || (*pqglob)->qprocess->fhdb)
    {
      iret = uuconf_hdb_init (ppglobal, zprogram);
      if (iret != UUCONF_SUCCESS)
	return iret;
    }
#endif

  return iret;
}
