/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 27, 2022.
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
#include "system.h"
#include "sysdep.h"

#include <errno.h>

#if HAVE_TIME_H
#include <time.h>
#endif

#if HAVE_FCNTL_H
#include <fcntl.h>
#else
#if HAVE_SYS_FILE_H
#include <sys/file.h>
#endif
#endif

#ifndef O_RDONLY
#define O_RDONLY 0
#define O_WRONLY 1
#define O_RDWR 2
#endif

#ifndef O_NOCTTY
#define O_NOCTTY 0
#endif

#ifndef FD_CLOEXEC
#define FD_CLOEXEC 1
#endif

#ifndef time
extern time_t time ();
#endif

/* Open a file to send to another system, and return the mode and
   the size.  */

openfile_t
esysdep_open_send (qsys, zfile, fcheck, zuser)
     const struct uuconf_system *qsys ATTRIBUTE_UNUSED;
     const char *zfile;
     boolean fcheck;
     const char *zuser;
{
  struct stat s;
  openfile_t e;
  int o;
  
  if (fsysdep_directory (zfile))
    {
      ulog (LOG_ERROR, "%s: is a directory", zfile);
      return EFILECLOSED;
    }

#if USE_STDIO
  e = fopen (zfile, BINREAD);
  if (e == NULL)
    {
      ulog (LOG_ERROR, "fopen (%s): %s", zfile, strerror (errno));
      return NULL;
    }
  o = fileno (e);
#else
  e = open ((char *) zfile, O_RDONLY | O_NOCTTY, 0);
  if (e == -1)
    {
      ulog (LOG_ERROR, "open (%s): %s", zfile, strerror (errno));
      return -1;
    }
  o = e;
#endif

  if (fcntl (o, F_SETFD, fcntl (o, F_GETFD, 0) | FD_CLOEXEC) < 0)
    {
      ulog (LOG_ERROR, "fcntl (FD_CLOEXEC): %s", strerror (errno));
      (void) ffileclose (e);
      return EFILECLOSED;
    }

  if (fstat (o, &s) == -1)
    {
      ulog (LOG_ERROR, "fstat: %s", strerror (errno));
      s.st_mode = 0666;
    }

  /* We have to recheck the file permission, although we probably
     checked it already, because otherwise there would be a window in
     which somebody could change the contents of a symbolic link to
     point to some file which was only readable by uucp.  */
  if (fcheck)
    {
      if (! fsuser_access (&s, R_OK, zuser))
	{
	  ulog (LOG_ERROR, "%s: %s", zfile, strerror (EACCES));
	  (void) ffileclose (e);
	  return EFILECLOSED;
	}
    }

  return e;
}

/* Get a temporary file name to receive into.  We use the ztemp
   argument to pick the file name, so that we restart the file if the
   transmission is aborted.  */

char *
zsysdep_receive_temp (qsys, zto, ztemp, frestart)
     const struct uuconf_system *qsys;
     const char *zto ATTRIBUTE_UNUSED;
     const char *ztemp;
     boolean frestart;
{
  if (frestart
      && ztemp != NULL
      && *ztemp == 'D'
      && strcmp (ztemp, "D.0") != 0)
    return zsappend3 (".Temp", qsys->uuconf_zname, ztemp);
  else
    return zstemp_file (qsys);
}  

/* The number of seconds in one week.  We must cast to long for this
   to be calculated correctly on a machine with 16 bit ints.  */
#define SECS_PER_WEEK ((long) 7 * (long) 24 * (long) 60 * (long) 60)

/* Open a temporary file to receive into.  This should, perhaps, check
   that we have write permission on the receiving directory, but it
   doesn't.  */

openfile_t
esysdep_open_receive (qsys, zto, ztemp, zreceive, pcrestart)
     const struct uuconf_system *qsys ATTRIBUTE_UNUSED;
     const char *zto ATTRIBUTE_UNUSED;
     const char *ztemp;
     const char *zreceive;
     long *pcrestart;
{
  int o;
  openfile_t e;

  /* If we used the ztemp argument in zsysdep_receive_temp, above,
     then we will have a name consistent across conversations.  In
     that case, we may have already received some portion of this
     file.  */
  o = -1;
  if (pcrestart != NULL)
    *pcrestart = -1;
  if (pcrestart != NULL
      && ztemp != NULL
      && *ztemp == 'D'
      && strcmp (ztemp, "D.0") != 0)
    {
      o = open ((char *) zreceive, O_WRONLY);
      if (o >= 0)
	{
	  struct stat s;

	  /* For safety, we insist on the file being less than 1 week
	     old.  This can still catch people, unfortunately.  I
	     don't know of any good solution to the problem of old
	     files hanging around.  If anybody has a file they want
	     restarted, and they know about this issue, they can touch
	     it to bring it up to date.  */
	  if (fstat (o, &s) < 0
	      || s.st_mtime + SECS_PER_WEEK < time ((time_t *) NULL))
	    {
	      (void) close (o);
	      o = -1;
	    }
	  else
	    {
	      DEBUG_MESSAGE1 (DEBUG_SPOOLDIR,
			      "esysdep_open_receive: Reusing %s",
			      zreceive);
	      *pcrestart = (long) s.st_size;
	    }
	}
    }

  if (o < 0)
    o = creat ((char *) zreceive, IPRIVATE_FILE_MODE);

  if (o < 0)
    {
      if (errno == ENOENT)
	{
	  if (! fsysdep_make_dirs (zreceive, FALSE))
	    return EFILECLOSED;
	  o = creat ((char *) zreceive, IPRIVATE_FILE_MODE);
	}
      if (o < 0)
	{
	  ulog (LOG_ERROR, "creat during esysdep_open_receive (%s): %s", zreceive, strerror (errno));
	  return EFILECLOSED;
	}
    }

  if (fcntl (o, F_SETFD, fcntl (o, F_GETFD, 0) | FD_CLOEXEC) < 0)
    {
      ulog (LOG_ERROR, "fcntl (FD_CLOEXEC): %s", strerror (errno));
      (void) close (o);
      (void) remove (zreceive);
      return EFILECLOSED;
    }

#if USE_STDIO
  e = fdopen (o, (char *) BINWRITE);

  if (e == NULL)
    {
      ulog (LOG_ERROR, "fdopen (%s): %s", zreceive, strerror (errno));
      (void) close (o);
      (void) remove (zreceive);
      return EFILECLOSED;
    }
#else
  e = o;
#endif

  return e;
}
