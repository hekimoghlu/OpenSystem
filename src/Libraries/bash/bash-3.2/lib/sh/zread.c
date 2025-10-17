/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 3, 2025.
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

#if !defined (errno)
extern int errno;
#endif

#ifndef SEEK_CUR
#  define SEEK_CUR 1
#endif

/* Read LEN bytes from FD into BUF.  Retry the read on EINTR.  Any other
   error causes the loop to break. */
ssize_t
zread (fd, buf, len)
     int fd;
     char *buf;
     size_t len;
{
  ssize_t r;

  while ((r = read (fd, buf, len)) < 0 && errno == EINTR)
    ;
  return r;
}

/* Read LEN bytes from FD into BUF.  Retry the read on EINTR, up to three
   interrupts.  Any other error causes the loop to break. */

#ifdef NUM_INTR
#  undef NUM_INTR
#endif
#define NUM_INTR 3

ssize_t
zreadintr (fd, buf, len)
     int fd;
     char *buf;
     size_t len;
{
  ssize_t r;
  int nintr;

  for (nintr = 0; ; )
    {
      r = read (fd, buf, len);
      if (r >= 0)
	return r;
      if (r == -1 && errno == EINTR)
	{
	  if (++nintr > NUM_INTR)
	    return -1;
	  continue;
	}
      return r;
    }
}

/* Read one character from FD and return it in CP.  Return values are as
   in read(2).  This does some local buffering to avoid many one-character
   calls to read(2), like those the `read' builtin performs. */

static char lbuf[128];
static size_t lind, lused;

ssize_t
zreadc (fd, cp)
     int fd;
     char *cp;
{
  ssize_t nr;

  if (lind == lused || lused == 0)
    {
      nr = zread (fd, lbuf, sizeof (lbuf));
      lind = 0;
      if (nr <= 0)
	{
	  lused = 0;
	  return nr;
	}
      lused = nr;
    }
  if (cp)
    *cp = lbuf[lind++];
  return 1;
}

void
zreset ()
{
  lind = lused = 0;
}

/* Sync the seek pointer for FD so that the kernel's idea of the last char
   read is the last char returned by zreadc. */
void
zsyncfd (fd)
     int fd;
{
  off_t off;
  int r;

  off = lused - lind;
  r = 0;
  if (off > 0)
    r = lseek (fd, -off, SEEK_CUR);

  if (r >= 0)
    lused = lind = 0;
}
