/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 24, 2022.
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

#if HAVE_FTW_H
#include <ftw.h>
#endif

static int isremove_dir P((const char *, const struct stat *, int));

/* Keep a list of directories to be removed.  */

struct sdirlist
{
  struct sdirlist *qnext;
  char *zdir;
};

static struct sdirlist *qSdirlist;

/* Remove a directory and all files in it.  */

boolean
fsysdep_rmdir (zdir)
     const char *zdir;
{
  boolean fret;
  struct sdirlist *q;

  qSdirlist = NULL;

  fret = TRUE;
  if (ftw ((char *) zdir, isremove_dir, 5) != 0)
    {
      ulog (LOG_ERROR, "ftw: %s", strerror (errno));
      fret = FALSE;
    }

  q = qSdirlist;
  while (q != NULL)
    {
      struct sdirlist *qnext;
      
      if (rmdir (q->zdir) != 0)
	{
	  ulog (LOG_ERROR, "rmdir (%s): %s", q->zdir, strerror (errno));
	  fret = FALSE;
	}
      ubuffree (q->zdir);
      qnext = q->qnext;
      xfree ((pointer) q);
      q = qnext;
    }

  return fret;
}

/* Remove a file in a directory.  */

/*ARGSUSED*/
static int
isremove_dir (zfile, qstat, iflag)
     const char *zfile;
     const struct stat *qstat ATTRIBUTE_UNUSED;
     int iflag;
{
  if (iflag == FTW_D || iflag == FTW_DNR)
    {
      struct sdirlist *q;

      q = (struct sdirlist *) xmalloc (sizeof (struct sdirlist));
      q->qnext = qSdirlist;
      q->zdir = zbufcpy (zfile);
      qSdirlist = q;
    }
  else
    {
      if (remove (zfile) != 0)
	ulog (LOG_ERROR, "remove (%s): %s", zfile, strerror (errno));
    }

  return 0;
}
