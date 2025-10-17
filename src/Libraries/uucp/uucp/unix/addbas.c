/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 18, 2023.
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

/* If we have a directory, add a base name.  */

char *
zsysdep_add_base (zfile, zname)
     const char *zfile;
     const char *zname;
{
  size_t clen;
  const char *zlook;
  char *zfree;
  char *zret;

#if DEBUG > 0
  if (*zfile != '/')
    ulog (LOG_FATAL, "zsysdep_add_base: %s: Can't happen", zfile);
#endif

  clen = strlen (zfile);

  if (zfile[clen - 1] != '/')
    {
      if (! fsysdep_directory (zfile))
	return zbufcpy (zfile);
      zfree = NULL;
    }
  else
    {
      /* Trim out the trailing '/'.  */
      zfree = zbufcpy (zfile);
      zfree[clen - 1] = '\0';
      zfile = zfree;
    }

  zlook = strrchr (zname, '/');
  if (zlook != NULL)
    zname = zlook + 1;

  zret = zsysdep_in_dir (zfile, zname);
  ubuffree (zfree);
  return zret;
}
