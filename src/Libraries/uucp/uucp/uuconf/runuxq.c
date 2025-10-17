/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 6, 2022.
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
const char _uuconf_runuxq_rcsid[] = "$Id: runuxq.c,v 1.6 2002/03/05 19:10:42 ian Rel $";
#endif

/* Return how often to spawn a uuxqt process.  This is either a
   positive number representing the number of execution files to be
   received between spawns, or a special code.  When using
   TAYLOR_CONFIG, this is from the ``run-uuxqt'' command in config
   (the default is UUCONF_RUNUUXQT_ONCE, for compatibility).
   Otherwise, we return UUCONF_RUNUUXQT_PERCALL for HDB_CONFIG and 10
   for V2_CONFIG, to emulate traditional HDB and V2 emulations.  */

int
uuconf_runuuxqt (pglobal, pirunuuxqt)
     pointer pglobal;
     int *pirunuuxqt;
{
#if HAVE_TAYLOR_CONFIG
  {
    struct sglobal *qglobal = (struct sglobal *) pglobal;
    const char *zrun;

    zrun = qglobal->qprocess->zrunuuxqt;
    if (zrun == NULL
	|| strcasecmp (zrun, "once") == 0)
      *pirunuuxqt = UUCONF_RUNUUXQT_ONCE;
    else if (strcasecmp (zrun, "never") == 0)
      *pirunuuxqt = UUCONF_RUNUUXQT_NEVER;
    else if (strcasecmp (zrun, "percall") == 0)
      *pirunuuxqt = UUCONF_RUNUUXQT_PERCALL;
    else
      {
	char *zend;

	*pirunuuxqt = strtol ((char *) qglobal->qprocess->zrunuuxqt,
			      &zend, 10);
	if (*zend != '\0' || *pirunuuxqt <= 0)
	  *pirunuuxqt = UUCONF_RUNUUXQT_ONCE;
      }
  }
#else /* ! HAVE_TAYLOR_CONFIG */
#if HAVE_HDB_CONFIG
  *pirunuuxqt = UUCONF_RUNUUXQT_PERCALL;
#else /* ! HAVE_HDB_CONFIG */
  *pirunuuxqt = 10;
#endif /* ! HAVE_HDB_CONFIG */
#endif /* ! HAVE_TAYLOR_CONFIG */

  return UUCONF_SUCCESS;
}
