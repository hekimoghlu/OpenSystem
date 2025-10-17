/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 14, 2024.
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

#if HAVE_TIME_H
#include <time.h>
#endif

#ifndef ctime
extern char *ctime ();
#endif

/* Mail a message to a user.  */

boolean
fsysdep_mail (zto, zsubject, cstrs, paz)
     const char *zto;
     const char *zsubject;
     int cstrs;
     const char **paz;
{
  char **pazargs;
  char *zcopy, *ztok;
  size_t cargs, iarg;
  FILE *e;
  pid_t ipid;
  time_t itime;
  int i;

  /* Parse MAIL_PROGRAM into an array of arguments.  */
  zcopy = zbufcpy (MAIL_PROGRAM);

  cargs = 0;
  for (ztok = strtok (zcopy, " \t");
       ztok != NULL;
       ztok = strtok ((char *) NULL, " \t"))
    ++cargs;

  pazargs = (char **) xmalloc ((cargs + 4) * sizeof (char *));

  memcpy (zcopy, MAIL_PROGRAM, sizeof MAIL_PROGRAM);
  for (ztok = strtok (zcopy, " \t"), iarg = 0;
       ztok != NULL;
       ztok = strtok ((char *) NULL, " \t"), ++iarg)
    pazargs[iarg] = ztok;

#if ! MAIL_PROGRAM_SUBJECT_BODY
  pazargs[iarg++] = (char *) "-s";
  pazargs[iarg++] = (char *) zsubject;
#endif

#if ! MAIL_PROGRAM_TO_BODY
  pazargs[iarg++] = (char *) zto;
#endif

  pazargs[iarg] = NULL;

  e = espopen ((const char **) pazargs, FALSE, &ipid);

  ubuffree (zcopy);
  xfree ((pointer) pazargs);

  if (e == NULL)
    {
      ulog (LOG_ERROR, "espopen (%s): %s", MAIL_PROGRAM,
	    strerror (errno));
      return FALSE;
    }

#if MAIL_PROGRAM_TO_BODY
  fprintf (e, "To: %s\n", zto);
#endif
#if MAIL_PROGRAM_SUBJECT_BODY
  fprintf (e, "Subject: %s\n", zsubject);
#endif

#if MAIL_PROGRAM_TO_BODY || MAIL_PROGRAM_SUBJECT_BODY
  fprintf (e, "\n");
#endif

  (void) time (&itime);
  /* Remember that ctime includes a \n, so this skips a line.  */
  fprintf (e, "Message from UUCP on %s %s\n", zSlocalname,
	   ctime (&itime));

  ulog(LOG_ERROR, "mail %s about %s on %s", zto, zsubject, zSlocalname);
  for (i = 0; i < cstrs; i++) {
    ulog(LOG_ERROR, "- %s", paz[i]);
    fputs (paz[i], e);
  }

  (void) fclose (e);

  return ixswait ((unsigned long) ipid, MAIL_PROGRAM) == 0;
}
