/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 8, 2024.
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

/* See whether running this file through zsysdep_add_cwd would require
   knowing the current working directory.  This is used to avoid
   determining the cwd if it will not be needed.  */

boolean
fsysdep_needs_cwd (zfile)
     const char *zfile;
{
  return *zfile != '/' && *zfile != '~';
}

/* Expand a local file, putting relative pathnames in the current
   working directory.  Note that ~/file is placed in the public
   directory, rather than in the user's home directory.  This is
   consistent with other UUCP packages.  */

char *
zsysdep_local_file_cwd (zfile, zpubdir, pfbadname)
     const char *zfile;
     const char *zpubdir;
     boolean *pfbadname;
{
  if (pfbadname != NULL)
    *pfbadname = FALSE;
  if (*zfile == '/')
    return zbufcpy (zfile);
  else if (*zfile == '~')
    return zsysdep_local_file (zfile, zpubdir, pfbadname);
  else
    return zsysdep_add_cwd (zfile);
}      

/* Add the current working directory to a remote file name.  */

char *
zsysdep_add_cwd (zfile)
     const char *zfile;
{
  if (*zfile == '/' || *zfile == '~')
    return zbufcpy (zfile);

  if (zScwd == NULL)
    {
      ulog (LOG_ERROR, "Can't determine current directory");
      return NULL;
    }

  return zsysdep_in_dir (zScwd, zfile);
}
