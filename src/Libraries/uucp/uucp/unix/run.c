/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 8, 2021.
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
#include "sysdep.h"
#include "system.h"

#include <errno.h>

/* Start up a new program.  */

boolean
fsysdep_run (ffork, zprogram, zarg1, zarg2)
     boolean ffork;
     const char *zprogram;
     const char *zarg1;
     const char *zarg2;
{
  char *zlib;
  const char *azargs[4];
  int aidescs[3];
  pid_t ipid;

  /* If we are supposed to fork, fork and then spawn so that we don't
     have to worry about zombie processes.  */
  if (ffork)
    {
      ipid = ixsfork ();
      if (ipid < 0)
	{
	  ulog (LOG_ERROR, "fork: %s", strerror (errno));
	  return FALSE;
	}

      if (ipid != 0)
	{
	  /* This is the parent.  Wait for the child we just forked to
	     exit (below) and return.  */
	  (void) ixswait ((unsigned long) ipid, (const char *) NULL);

	  /* Force the log files to be reopened in case the child just
	     output any error messages and stdio doesn't handle
	     appending correctly.  */
	  ulog_close ();

	  return TRUE;
	}

      /* This is the child.  Detach from the terminal to avoid any
	 unexpected SIGHUP signals.  At this point we are definitely
	 not a process group leader, so usysdep_detach will not fork
	 again.  */
      usysdep_detach ();

      /* Now spawn the program and then exit.  */
    }

  zlib = zbufalc (sizeof SBINDIR + sizeof "/" + strlen (zprogram));
  sprintf (zlib, "%s/%s", SBINDIR, zprogram);

  azargs[0] = zlib;
  azargs[1] = zarg1;
  azargs[2] = zarg2;
  azargs[3] = NULL;

  aidescs[0] = SPAWN_NULL;
  aidescs[1] = SPAWN_NULL;
  aidescs[2] = SPAWN_NULL;

  /* We pass fsetuid and fshell as TRUE, which permits uucico and
     uuxqt to be replaced by (non-setuid) shell scripts.  */
  ipid = ixsspawn (azargs, aidescs, TRUE, FALSE, (const char *) NULL,
		   FALSE, TRUE, (const char *) NULL,
		   (const char *) NULL, (const char *) NULL);
  ubuffree (zlib);

  if (ipid < 0)
    {
      ulog (LOG_ERROR, "ixsspawn: %s", strerror (errno));
      if (ffork)
	_exit (EXIT_FAILURE);
      return FALSE;
    }

  if (ffork)
    _exit (EXIT_SUCCESS);

  return TRUE;
}
