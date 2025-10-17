/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 18, 2024.
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
const char _uuconf_vinit_rcsid[] = "$Id: vinit.c,v 1.6 2002/03/05 19:10:43 ian Rel $";
#endif

#include <errno.h>

static int ivinlib P((struct sglobal *qglobal, const char *z, size_t csize,
		      char **pz));

/* Return an allocated buffer holding a file name in OLDCONFIGLIB.
   The c argument is the size of z including the trailing null byte,
   since this is convenient for both the caller and this function.  */

static int
ivinlib (qglobal, z, c, pz)
     struct sglobal *qglobal;
     const char *z;
     size_t c;
     char **pz;
{
  char *zalc;

  zalc = uuconf_malloc (qglobal->pblock, sizeof OLDCONFIGLIB - 1 + c);
  if (zalc == NULL)
    {
      qglobal->ierrno = errno;
      return UUCONF_MALLOC_FAILED | UUCONF_ERROR_ERRNO;
    }

  memcpy ((pointer) zalc, (pointer) OLDCONFIGLIB,
	  sizeof OLDCONFIGLIB - 1);
  memcpy ((pointer) (zalc + sizeof OLDCONFIGLIB - 1), (pointer) z, c);

  *pz = zalc;

  return UUCONF_SUCCESS;
}

/* Initialize the routines which read V2 configuration files.  The
   only thing we do here is allocate the file names.  */

int
uuconf_v2_init (ppglobal)
     pointer *ppglobal;
{
  struct sglobal **pqglobal = (struct sglobal **) ppglobal;
  int iret;
  struct sglobal *qglobal;
  char *zdialcodes;

  if (*pqglobal == NULL)
    {
      iret = _uuconf_iinit_global (pqglobal);
      if (iret != UUCONF_SUCCESS)
	return iret;
    }

  qglobal = *pqglobal;

  iret = ivinlib (qglobal, V2_SYSTEMS, sizeof V2_SYSTEMS,
		  &qglobal->qprocess->zv2systems);
  if (iret != UUCONF_SUCCESS)
    return iret;
  iret = ivinlib (qglobal, V2_DEVICES, sizeof V2_DEVICES,
		  &qglobal->qprocess->zv2devices);
  if (iret != UUCONF_SUCCESS)
    return iret;
  iret = ivinlib (qglobal, V2_USERFILE, sizeof V2_USERFILE,
		  &qglobal->qprocess->zv2userfile);
  if (iret != UUCONF_SUCCESS)
    return iret;
  iret = ivinlib (qglobal, V2_CMDS, sizeof V2_CMDS,
		  &qglobal->qprocess->zv2cmds);
  if (iret != UUCONF_SUCCESS)
    return iret;

  iret = ivinlib (qglobal, V2_DIALCODES, sizeof V2_DIALCODES,
		  &zdialcodes);
  if (iret != UUCONF_SUCCESS)
    return iret;

  return _uuconf_iadd_string (qglobal, zdialcodes, FALSE, FALSE,
			      &qglobal->qprocess->pzdialcodefiles,
			      qglobal->pblock);
}
