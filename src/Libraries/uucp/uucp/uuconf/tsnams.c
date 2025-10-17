/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 2, 2022.
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
const char _uuconf_tsnams_rcsid[] = "$Id: tsnams.c,v 1.6 2002/03/05 19:10:43 ian Rel $";
#endif

/* Get all the system names from the Taylor UUCP configuration files.
   These were actually already recorded by uuconf_taylor_init, so this
   function is pretty simple.  */

int
uuconf_taylor_system_names (pglobal, ppzsystems, falias)
     pointer pglobal;
     char ***ppzsystems;
     int falias;
{
  struct sglobal *qglobal = (struct sglobal *) pglobal;
  int iret;
  register struct stsysloc *q;
  char **pz;
  int c, i;

  if (! qglobal->qprocess->fread_syslocs)
    {
      iret = _uuconf_iread_locations (qglobal);
      if (iret != UUCONF_SUCCESS)
	return iret;
    }

  *ppzsystems = NULL;
  c = 0;

  for (q = qglobal->qprocess->qsyslocs; q != NULL; q = q->qnext)
    {
      if (! falias && q->falias)
	continue;

      iret = _uuconf_iadd_string (qglobal, (char *) q->zname, TRUE, FALSE,
				  ppzsystems, (pointer) NULL);
      if (iret != UUCONF_SUCCESS)
	return iret;
      ++c;
    }

  /* The order of the qSyslocs list is reversed from the list in the
     configuration files.  Reverse the returned list in order to make
     uuname output more intuitive.  */
  pz = *ppzsystems;
  for (i = c / 2 - 1; i >= 0; i--)
    {
      char *zhold;

      zhold = pz[i];
      pz[i] = pz[c - i - 1];
      pz[c - i - 1] = zhold;
    }

  return UUCONF_SUCCESS;
}
