/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 15, 2022.
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

unsigned int
ixsysdep_file_mode (zfile)
     const char *zfile;
{
  struct stat s;

  if (stat ((char *) zfile, &s) != 0)
    {
      ulog (LOG_ERROR, "stat (%s): %s", zfile, strerror (errno));
      return 0;
    }

#if S_IRWXU != 0700
 #error Files modes need to be translated
#endif

  /* We can't return 0, since that indicates an error.  */
  if ((s.st_mode & 0777) == 0)
    return 0400;

  return s.st_mode & 0777;
}
