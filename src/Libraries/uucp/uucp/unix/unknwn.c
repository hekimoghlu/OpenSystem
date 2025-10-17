/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 15, 2024.
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

/* Run the remote.unknown shell script.  If it succeeds, we return
   FALSE because that means that the system is not permitted to log
   in.  If the execution fails, we return TRUE.  */

boolean
fsysdep_unknown_caller (zscript, zsystem)
     const char *zscript;
     const char *zsystem;
{
  const char *azargs[3];
  int aidescs[3];
  pid_t ipid;

  azargs[0] = zscript;
  azargs[1] = zsystem;
  azargs[2] = NULL;

  aidescs[0] = SPAWN_NULL;
  aidescs[1] = SPAWN_NULL;
  aidescs[2] = SPAWN_NULL;

  ipid = ixsspawn (azargs, aidescs, TRUE, TRUE, (const char *) NULL, FALSE,
		   TRUE, (const char *) NULL, (const char *) NULL,
		   (const char *) NULL);
  if (ipid < 0)
    {
      ulog (LOG_ERROR, "ixsspawn: %s", strerror (errno));
      return FALSE;
    }

  return ixswait ((unsigned long) ipid, (const char *) NULL) != 0;
}
