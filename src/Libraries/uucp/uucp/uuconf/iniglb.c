/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 18, 2024.
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
const char _uuconf_iniglb_rcsid[] = "$Id: iniglb.c,v 1.14 2002/03/05 19:10:42 ian Rel $";
#endif

#include <errno.h>

/* Initialize the global information structure.  */

int
_uuconf_iinit_global (pqglobal)
     struct sglobal **pqglobal;
{
  pointer pblock;
  register struct sprocess *qprocess;
  char *azargs[3];
  int iret;

  pblock = uuconf_malloc_block ();
  if (pblock == NULL)
    return UUCONF_MALLOC_FAILED;

  *pqglobal = (struct sglobal *) uuconf_malloc (pblock,
						sizeof (struct sglobal));
  if (*pqglobal == NULL)
    {
      uuconf_free_block (pblock);
      return UUCONF_MALLOC_FAILED;
    }

  (*pqglobal)->qprocess = ((struct sprocess *)
			   uuconf_malloc (pblock,
					  sizeof (struct sprocess)));
  if ((*pqglobal)->qprocess == NULL)
    {
      uuconf_free_block (pblock);
      *pqglobal = NULL;
      return UUCONF_MALLOC_FAILED;
    }

  (*pqglobal)->pblock = pblock;
  (*pqglobal)->ierrno = 0;
  (*pqglobal)->ilineno = 0;
  (*pqglobal)->zfilename = NULL;

  qprocess = (*pqglobal)->qprocess;

  qprocess->zlocalname = NULL;
  qprocess->zspooldir = SPOOLDIR;
  qprocess->zpubdir = PUBDIR;
#ifdef LOCKDIR
  qprocess->zlockdir = LOCKDIR;
#else
  qprocess->zlockdir = SPOOLDIR;
#endif
  qprocess->zlogfile = LOGFILE;
  qprocess->zstatsfile = STATFILE;
  qprocess->zdebugfile = DEBUGFILE;
  qprocess->zdebug = "";
  qprocess->fstrip_login = TRUE;
  qprocess->fstrip_proto = TRUE;
  qprocess->cmaxuuxqts = 0;
  qprocess->zrunuuxqt = NULL;
  qprocess->fv2 = TRUE;
  qprocess->fhdb = TRUE;
  qprocess->pzdialcodefiles = NULL;
  qprocess->pztimetables = NULL;
  qprocess->zconfigfile = NULL;
  qprocess->pzsysfiles = NULL;
  qprocess->pzportfiles = NULL;
  qprocess->pzdialfiles = NULL;
  qprocess->pzpwdfiles = NULL;
  qprocess->pzcallfiles = NULL;
  qprocess->qunknown = NULL;
  qprocess->fread_syslocs = FALSE;
  qprocess->qsyslocs = NULL;
  qprocess->qvalidate = NULL;
  qprocess->fuses_myname = FALSE;
  qprocess->zv2systems = NULL;
  qprocess->zv2devices = NULL;
  qprocess->zv2userfile = NULL;
  qprocess->zv2cmds = NULL;
  qprocess->pzhdb_systems = NULL;
  qprocess->pzhdb_devices = NULL;
  qprocess->pzhdb_dialers = NULL;
  qprocess->fhdb_read_permissions = FALSE;
  qprocess->qhdb_permissions = NULL;

  azargs[0] = NULL;
  azargs[1] = (char *) "Evening";
  azargs[2] = (char *) "Wk1705-0755,Sa,Su";
  iret = _uuconf_itimetable ((pointer) *pqglobal, 3, azargs,
			     (pointer) NULL, (pointer) NULL);
  if (UUCONF_ERROR_VALUE (iret) == UUCONF_SUCCESS)
    {
      azargs[1] = (char *) "Night";
      azargs[2] = (char *) "Wk2305-0755,Sa,Su2305-1655";
      iret = _uuconf_itimetable ((pointer) *pqglobal, 3, azargs,
				 (pointer) NULL, (pointer) NULL);
    }
  if (UUCONF_ERROR_VALUE (iret) == UUCONF_SUCCESS)
    {
      azargs[1] = (char *) "NonPeak";
      azargs[2] = (char *) "Wk1805-0655,Sa,Su";
      iret = _uuconf_itimetable ((pointer) *pqglobal, 3, azargs,
				 (pointer) NULL, (pointer) NULL);
    }
  if (UUCONF_ERROR_VALUE (iret) != UUCONF_SUCCESS)
    {
      uuconf_free_block (pblock);
      *pqglobal = NULL;

      /* Strip off any special bits, since there's no global
	 structure.  */
      return UUCONF_ERROR_VALUE (iret);
    }

  return UUCONF_SUCCESS;
}

/* Add a timetable.  This is also called by the Taylor UUCP
   initialization code, as well as by the Taylor UUCP sys file code
   (although the latter is obsolete).  There's no point in putting
   this in a separate file, since everything must call
   _uuconf_init_global.  There is a race condition here if this is
   called by two different threads on a sys file command, but the sys
   file command is obsolete anyhow.  */

/*ARGSUSED*/
int
_uuconf_itimetable (pglobal, argc, argv, pvar, pinfo)
     pointer pglobal;
     int argc ATTRIBUTE_UNUSED;
     char **argv;
     pointer pvar ATTRIBUTE_UNUSED;
     pointer pinfo ATTRIBUTE_UNUSED;
{
  struct sglobal *qglobal = (struct sglobal *) pglobal;
  int iret;

  iret = _uuconf_iadd_string (qglobal, argv[1], FALSE, FALSE,
			      &qglobal->qprocess->pztimetables,
			      qglobal->pblock);
  if (iret != UUCONF_SUCCESS)
    return iret | UUCONF_CMDTABRET_EXIT;

  iret = _uuconf_iadd_string (qglobal, argv[2], FALSE, FALSE,
			      &qglobal->qprocess->pztimetables,
			      qglobal->pblock);
  if (iret != UUCONF_SUCCESS)
    return iret | UUCONF_CMDTABRET_EXIT;

  return UUCONF_CMDTABRET_KEEP;
}
