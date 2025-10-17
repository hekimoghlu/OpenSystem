/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 22, 2025.
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

/* Open a file with the permissions of the invoking user.  Ignore the
   fbinary argument since Unix has no distinction between text and
   binary files.  */

/*ARGSUSED*/
openfile_t
esysdep_user_fopen (zfile, frd, fbinary)
     const char *zfile;
     boolean frd;
     boolean fbinary ATTRIBUTE_UNUSED;
{
  uid_t ieuid;
  gid_t iegid;
  openfile_t e;
  const char *zerr;
  int o = 0;

  if (! fsuser_perms (&ieuid, &iegid))
    return EFILECLOSED;

  zerr = NULL;

#if USE_STDIO
  e = fopen (zfile, frd ? "r" : "w");
  if (e == NULL)
    zerr = "fopen";
  else
    o = fileno (e);
#else
  if (frd)
    {
      e = open ((char *) zfile, O_RDONLY | O_NOCTTY, 0);
      zerr = "open";
    }
  else
    {
      e = creat ((char *) zfile, IPUBLIC_FILE_MODE);
      zerr = "creat";
    }
  if (e >= 0)
    {
      o = e;
      zerr = NULL;
    }
#endif

  if (! fsuucp_perms ((long) ieuid, (long) iegid))
    {
      if (ffileisopen (e))
	(void) ffileclose (e);
      return EFILECLOSED;
    }

  if (zerr != NULL)
    {
      ulog (LOG_ERROR, "%s (%s): %s", zerr, zfile, strerror (errno));
#if ! HAVE_SETREUID
      /* Are these error messages helpful or confusing?  */
#if HAVE_SAVED_SETUID
      if (errno == EACCES && getuid () == 0)
	ulog (LOG_ERROR,
	      "The superuser may only transfer files that are readable by %s",
	      OWNER);
#else
      if (errno == EACCES)
	ulog (LOG_ERROR,
	      "You may only transfer files that are readable by %s", OWNER);
#endif
#endif /* ! HAVE_SETREUID */
      return EFILECLOSED;
    }

  if (fcntl (o, F_SETFD, fcntl (o, F_GETFD, 0) | FD_CLOEXEC) < 0)
    {
      ulog (LOG_ERROR, "fcntl (FD_CLOEXEC): %s", strerror (errno));
      (void) ffileclose (e);
      return EFILECLOSED;
    }

  return e;
}
