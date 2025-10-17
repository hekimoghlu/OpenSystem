/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 29, 2021.
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

#if HAVE_FTW_H
#include <ftw.h>
#endif

static int iswalk_dir P((const char *zname, const struct stat *qstat,
			 int iflag));

/* Walk a directory tree.  */

static size_t cSlen;
static void (*puSfn) P((const char *zfull, const char *zrelative,
			pointer pinfo));
static pointer pSinfo;

boolean
usysdep_walk_tree (zdir, pufn, pinfo)
     const char *zdir;
     void (*pufn) P((const char *zfull, const char *zrelative,
		     pointer pinfo));
     pointer pinfo;
{
  cSlen = strlen (zdir) + 1;
  puSfn = pufn;
  pSinfo = pinfo;
  return ftw ((char *) zdir, iswalk_dir, 5) == 0;
}

/* Pass a file found in the directory tree to the system independent
   function.  */

/*ARGSUSED*/
static int
iswalk_dir (zname, qstat, iflag)
     const char *zname;
     const struct stat *qstat ATTRIBUTE_UNUSED;
     int iflag;
{
  char *zcopy;

  if (iflag != FTW_F)
    return 0;

  zcopy = zbufcpy (zname + cSlen);

  (*puSfn) (zname, zcopy, pSinfo);

  ubuffree (zcopy);

  return 0;
}
