/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 1, 2023.
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

#include "sysdep.h"

#include <errno.h>

/* A version of popen that goes through ixsspawn.  This actually takes
   an array of arguments rather than a string, and takes a boolean
   read/write value rather than a string.  It sets *pipid to the
   process ID of the child.  */

FILE *
espopen (pazargs, frd, pipid)
     const char **pazargs;
     boolean frd;
     pid_t *pipid;
{
  int aidescs[3];
  pid_t ipid;
  FILE *eret;

  if (frd)
    {
      aidescs[0] = SPAWN_NULL;
      aidescs[1] = SPAWN_READ_PIPE;
    }
  else
    {
      aidescs[0] = SPAWN_WRITE_PIPE;
      aidescs[1] = SPAWN_NULL;
    }
  aidescs[2] = SPAWN_NULL;

  ipid = ixsspawn (pazargs, aidescs, TRUE, FALSE,
		   (const char *) NULL, FALSE, TRUE,
		   (const char *) NULL, (const char *) NULL,
		   (const char *) NULL);
  if (ipid < 0)
    return NULL;

  if (frd)
    eret = fdopen (aidescs[1], (char *) "r");
  else
    eret = fdopen (aidescs[0], (char *) "w");
  if (eret == NULL)
    {
      int ierr;

      ierr = errno;
      (void) close (frd ? aidescs[1] : aidescs[0]);
      (void) kill (ipid, SIGKILL);
      (void) ixswait ((unsigned long) ipid, (const char *) NULL);
      errno = ierr;
      return NULL;
    }
    
  *pipid = ipid;

  return eret;
}
