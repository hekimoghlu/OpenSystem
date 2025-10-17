/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 18, 2021.
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

#include <pwd.h>

#if GETPWNAM_DECLARATION_OK
#ifndef getpwnam
extern struct passwd *getpwnam ();
#endif
#endif

/* Turn a file name into an absolute path, by doing tilde expansion
   and moving any other type of file into the public directory.  */

char *
zsysdep_local_file (zfile, zpubdir, pfbadname)
     const char *zfile;
     const char *zpubdir;
     boolean *pfbadname;
{
  const char *zdir;

  if (pfbadname != NULL)
    *pfbadname = FALSE;

  if (*zfile == '/')
    return zbufcpy (zfile);

  if (*zfile != '~')
    zdir = zpubdir;
  else
    {
      if (zfile[1] == '\0')
	return zbufcpy (zpubdir);

      if (zfile[1] == '/')
	{
	  zdir = zpubdir;
	  zfile += 2;
	}
      else
	{
#if defined(ALL_USERS_ARE_PUBDIR)
	  zdir = zpubdir;
	  zfile += strcspn((char *)zfile, "/");
	  if (*zfile) {
	      zfile += 1;
	  }
#else
	  size_t cuserlen;
	  char *zcopy, *ztmp;
	  struct passwd *q;

	  ++zfile;
	  cuserlen = strcspn ((char *) zfile, "/");
	  zcopy = zbufalc (cuserlen + 1);
	  memcpy (zcopy, zfile, cuserlen);
	  zcopy[cuserlen] = '\0';
      
	  q = getpwnam (zcopy);
	  if (q == NULL)
	    {
	      /* We can't log this, it causes us to fail a test :-( */
	      /* ulog (LOG_DEBUG, "User %s not found, using pubdir (%s)",
	        zcopy, zpubdir); */
	      ztmp = zpubdir;
	    } else {
	      ztmp = q->pw_dir;
	    }
	  ubuffree (zcopy);

	  if (zfile[cuserlen] == '\0')
	    return zbufcpy(ztmp);

	  zdir = ztmp;
	  zfile += cuserlen + 1;
#endif
	}
    }

  return zsysdep_in_dir (zdir, zfile);
}
