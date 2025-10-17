/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 30, 2022.
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
#include <config.h>

#include <sys/types.h>

#if defined (HAVE_UNISTD_H)
#  include <unistd.h>
#endif

#include <errno.h>

#include <stdc.h>

#if !defined (errno)
extern int errno;
#endif

extern ssize_t zread __P((int, char *, size_t));
extern int zwrite __P((int, char *, ssize_t));

/* Dump contents of file descriptor FD to OFD.  FN is the filename for
   error messages (not used right now). */
int
zcatfd (fd, ofd, fn)
     int fd, ofd;
     char *fn;
{
  ssize_t nr;
  int rval;
  char lbuf[128];

  rval = 0;
  while (1)
    {
      nr = zread (fd, lbuf, sizeof (lbuf));
      if (nr == 0)
	break;
      else if (nr < 0)
	{
	  rval = -1;
	  break;
	}
      else if (zwrite (ofd, lbuf, nr) < 0)
	{
	  rval = -1;
	  break;
	}
    }

  return rval;
}
