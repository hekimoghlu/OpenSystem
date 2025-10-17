/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 21, 2023.
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
const char _uuconf_maxuxq_rcsid[] = "$Id: maxuxq.c,v 1.7 2002/03/05 19:10:42 ian Rel $";
#endif

/* Get the maximum number of simultaneous uuxqt executions.  When
   using TAYLOR_CONFIG, this is from the ``max-uuxqts'' command in
   config.  Otherwise, when using HDB_CONFIG, we read the file
   Maxuuxqts.  */

int
uuconf_maxuuxqts (pglobal, pcmax)
     pointer pglobal;
     int *pcmax;
{
#if HAVE_TAYLOR_CONFIG
  {
    struct sglobal *qglobal = (struct sglobal *) pglobal;

    *pcmax = qglobal->qprocess->cmaxuuxqts;
    return UUCONF_SUCCESS;
  }
#else /* ! HAVE_TAYLOR_CONFIG */
#if HAVE_HDB_CONFIG
  {
    char ab[sizeof OLDCONFIGLIB + sizeof HDB_MAXUUXQTS - 1];
    FILE *e;

    *pcmax = 0;

    memcpy ((pointer) ab, (constpointer) OLDCONFIGLIB,
	    sizeof OLDCONFIGLIB - 1);
    memcpy ((pointer) (ab + sizeof OLDCONFIGLIB - 1),
	    (constpointer) HDB_MAXUUXQTS, sizeof HDB_MAXUUXQTS);
    e = fopen (ab, "r");
    if (e != NULL)
      {
	char *z;
	size_t c;

	z = NULL;
	c = 0;
	if (getline (&z, &c, e) > 0)
	  {
	    *pcmax = (int) strtol (z, (char **) NULL, 10);
	    if (*pcmax < 0)
	      *pcmax = 0;
	    free ((pointer) z);
	  }
	(void) fclose (e);
      }

    return UUCONF_SUCCESS;
  }
#else /* ! HAVE_HDB_CONFIG */
  *pcmax = 0;
  return UUCONF_SUCCESS;
#endif /* ! HAVE_HDB_CONFIG */
#endif /* ! HAVE_TAYLOR_CONFIG */
}
