/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 4, 2025.
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

/* Get the current conversation sequence number for a remote system,
   and increment it for next time.  The conversation sequence number
   is kept in a file named for the system in the directory .Sequence
   in the spool directory.  This is not compatible with other versions
   of UUCP, but it makes more sense to me.  The sequence file is only
   used if specified in the information for that system.  */

long
ixsysdep_get_sequence (qsys)
     const struct uuconf_system *qsys;
{
  FILE *e;
  char *zname;
  struct stat s;
  long iseq;

  /* This will only be called when the system is locked anyhow, so there
     is no need to use a separate lock for the conversation sequence
     file.  */
#if SPOOLDIR_HDB || SPOOLDIR_SVR4
  zname = zsysdep_in_dir (".SQFILE", qsys->uuconf_zname);
#else
  zname = zsysdep_in_dir (".Sequence", qsys->uuconf_zname);
#endif

  iseq = 0;
  if (stat (zname, &s) == 0)
    {
      boolean fok;
      char *zline;
      size_t cline;

      /* The file should only be readable and writable by uucp.  */
      if ((s.st_mode & (S_IRWXG | S_IRWXO)) != 0)
	{
	  ulog (LOG_ERROR,
		"Bad file protection for conversation sequence file");
	  ubuffree (zname);
	  return -1;
	}
    
      e = fopen (zname, "r+");
      if (e == NULL)
	{
	  ulog (LOG_ERROR, "fopen (%s): %s", zname, strerror (errno));
	  ubuffree (zname);
	  return -1;
	}

      ubuffree (zname);

      fok = TRUE;
      zline = NULL;
      cline = 0;
      if (getline (&zline, &cline, e) <= 0)
	fok = FALSE;
      else
	{
	  char *zend;

	  iseq = strtol (zline, &zend, 10);
	  if (zend == zline)
	    fok = FALSE;
	}

      xfree ((pointer) zline);

      if (! fok)
	{
	  ulog (LOG_ERROR, "Bad format for conversation sequence file");
	  (void) fclose (e);
	  return -1;
	}

      rewind (e);
    }
  else
    {
      e = esysdep_fopen (zname, FALSE, FALSE, TRUE);
      ubuffree (zname);
      if (e == NULL)
	return -1;
    }

  ++iseq;

  fprintf (e, "%ld", iseq);

  if (fclose (e) != 0)
    {
      ulog (LOG_ERROR, "fclose: %s", strerror (errno));
      return -1;
    }

  return iseq;
}
