/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 19, 2025.
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
/* Copyright (C) 1987, 1991 Free Software Foundation, Inc.

   This file is part of GNU Bash, the Bourne Again SHell.

   Bash is free software; you can redistribute it and/or modify it
   under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 2, or (at your option)
   any later version.

   Bash is distributed in the hope that it will be useful, but WITHOUT
   ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
   or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public
   License for more details.

   You should have received a copy of the GNU General Public License
   along with Bash; see the file COPYING.  If not, write to the Free
   Software Foundation, 59 Temple Place, Suite 330, Boston, MA 02111 USA. */

/*  Write output in 128-byte chunks until we get a sigpipe or write gets an
    EPIPE.  Then report how many bytes we wrote.  We assume that this is the
    pipe size. */
#include <config.h>

#if defined (HAVE_UNISTD_H)
#  ifdef _MINIX
#    include <sys/types.h>
#  endif
#  include <unistd.h>
#endif

#include <stdio.h>
#ifndef _MINIX
#include "../bashtypes.h"
#endif
#include <signal.h>
#include <errno.h>

#include "../command.h"
#include "../general.h"
#include "../sig.h"

#ifndef errno
extern int errno;
#endif

int nw;

sighandler
sigpipe (sig)
     int sig;
{
  fprintf (stderr, "%d\n", nw);
  exit (0);
}

int
main (argc, argv)
     int argc;
     char **argv;
{
  char buf[128];
  register int i;

  for (i = 0; i < 128; i++)
    buf[i] = ' ';

  signal (SIGPIPE, sigpipe);

  nw = 0;
  for (;;)
    {
      int n;
      n = write (1, buf, 128);
      nw += n;
    }
  return (0);
}
