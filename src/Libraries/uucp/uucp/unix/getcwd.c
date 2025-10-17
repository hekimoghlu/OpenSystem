/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 23, 2024.
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

char *
getcwd (zbuf, cbuf)
     char *zbuf;
     size_t cbuf;
{
  const char *azargs[2];
  FILE *e;
  pid_t ipid;
  int cread;
  int ierr;

  azargs[0] = PWD_PROGRAM;
  azargs[1] = NULL;
  e = espopen (azargs, TRUE, &ipid);
  if (e == NULL)
    return NULL;

  ierr = 0;

  cread = fread (zbuf, sizeof (char), cbuf, e);
  if (cread == 0)
    ierr = errno;

  (void) fclose (e);

  if (ixswait ((unsigned long) ipid, (const char *) NULL) != 0)
    {
      ierr = EACCES;
      cread = 0;
    }

  if (cread != 0)
    {
      if (zbuf[cread - 1] == '\n')
	zbuf[cread - 1] = '\0';
      else
	{
	  ierr = ERANGE;
	  cread = 0;
	}
    }
  
  if (cread == 0)
    {
      errno = ierr;
      return NULL;
    }

  return zbuf;
}
