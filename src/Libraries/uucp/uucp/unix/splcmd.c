/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 21, 2023.
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
#include "uucp.h"

#include "uudefs.h"
#include "uuconf.h"
#include "sysdep.h"
#include "system.h"

#include <errno.h>
#include <ctype.h>

/* Given a set of commands to execute for a remote system, create a
   command file holding them.  This creates a single command file
   holding all the commands passed in.  It returns a jobid.  */

char *
zsysdep_spool_commands (qsys, bgrade, ccmds, pascmds, pftemp)
     const struct uuconf_system *qsys;
     int bgrade;
     int ccmds;
     const struct scmd *pascmds;
     boolean *pftemp;
{
  char abtempfile[sizeof "TMP1234567890"];
  char *ztemp;
  FILE *e;
  int i;
  const struct scmd *qcmd;
  char *z;
  char *zjobid;

  if (pftemp != NULL)
    *pftemp = TRUE;

#if DEBUG > 0
  if (! UUCONF_GRADE_LEGAL (bgrade))
    ulog (LOG_FATAL, "Bad grade %d", bgrade);
#endif

  /* Write the commands into a temporary file and then rename it to
     avoid a race with uucico reading the file.  */
  sprintf (abtempfile, "TMP%010lx", (unsigned long) getpid ());
  ztemp = zsfind_file (abtempfile, qsys->uuconf_zname, bgrade);
  if (ztemp == NULL)
    return NULL;

  e = esysdep_fopen (ztemp, FALSE, FALSE, TRUE);
  if (e == NULL)
    {
      ubuffree (ztemp);
      return NULL;
    }

  for (i = 0, qcmd = pascmds; i < ccmds; i++, qcmd++)
    {
      boolean fquote;
      const struct scmd *q;
      struct scmd squoted;

      fquote = fcmd_needs_quotes (qcmd);
      if (! fquote)
	q = qcmd;
      else
	{
	  uquote_cmd (qcmd, &squoted);
	  q = &squoted;
	}

      switch (q->bcmd)
	{
	case 'S':
	  fprintf (e, "S %s %s %s -%s %s 0%o %s\n", q->zfrom, q->zto,
		   q->zuser, q->zoptions, q->ztemp, q->imode,
		   q->znotify == NULL ? (const char *) "" : q->znotify);
	  break;
	case 'R':
	  fprintf (e, "R %s %s %s -%s\n", q->zfrom, q->zto, q->zuser,
		   q->zoptions);
	  break;
	case 'X':
	  fprintf (e, "X %s %s %s -%s\n", q->zfrom, q->zto, q->zuser,
		   q->zoptions);
	  break;
	case 'E':
	  fprintf (e, "E %s %s %s -%s %s 0%o %s 0 %s\n", q->zfrom, q->zto,
		   q->zuser, q->zoptions, q->ztemp, q->imode,
		   q->znotify, q->zcmd);
	  break;
	default:
	  ulog (LOG_ERROR,
		"zsysdep_spool_commands: Unrecognized type %d",
		q->bcmd);
	  (void) fclose (e);
	  (void) remove (ztemp);
	  ubuffree (ztemp);
	  if (pftemp != NULL)
	    *pftemp = FALSE;
	  return NULL;
	}

      if (fquote)
	ufree_quoted_cmd (&squoted);
    }

  if (! fstdiosync (e, ztemp))
    {
      (void) fclose (e);
      (void) remove (ztemp);
      ubuffree (ztemp);
      return NULL;
    }

  if (fclose (e) != 0)
    {
      ulog (LOG_ERROR, "fclose: %s", strerror (errno));
      (void) remove (ztemp);
      ubuffree (ztemp);
      return NULL;
    }

  /* The filename returned by zscmd_file is subject to some unlikely
     race conditions, so keep trying the link until the destination
     file does not already exist.  Each call to zscmd_file should
     return a file name which does not already exist, so we don't have
     to do anything special before calling it again.  */
  while (TRUE)
    {
      z = zscmd_file (qsys, bgrade);
      if (z == NULL)
	{
	  (void) remove (ztemp);
	  ubuffree (ztemp);
	  return NULL;
	}

      if (link (ztemp, z) >= 0)
	break;

      if (errno != EEXIST)
	{
	  ulog (LOG_ERROR, "link (%s, %s): %s", ztemp, z, strerror (errno));
	  (void) remove (ztemp);
	  ubuffree (ztemp);
	  ubuffree (z);
	  return NULL;
	}

      ubuffree (z);
    }

  (void) remove (ztemp);
  ubuffree (ztemp);

  zjobid = zsfile_to_jobid (qsys, z, bgrade);
  if (zjobid == NULL)
    (void) remove (z);
  ubuffree (z);
  return zjobid;
}
