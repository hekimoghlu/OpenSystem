/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 16, 2022.
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

/* Change the mode of a file.  */

boolean
fsysdep_change_mode (zfile, imode)
     const char *zfile;
     unsigned int imode;
{
  char rfile[PATH_MAX];

#ifdef WORLD_WRITABLE_FILE_IN
  realpath(zfile, rfile);
  if (rfile == strstr(rfile, WORLD_WRITABLE_FILE_IN)) {
      imode |= S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP | S_IROTH | S_IWOTH;
  }
#endif

  if (chmod ((char *) zfile, imode) < 0)
    {
      ulog (LOG_ERROR, "chmod (%s): %s", zfile, strerror (errno));
      return FALSE;
    }
  return TRUE;
}
