/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 6, 2022.
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
/* Copyright (C) 1995 Free Software Foundation, Inc.

   This file is part of GNU Bash, the Bourne Again SHell.

   Bash is free software; you can redistribute it and/or modify it under
   the terms of the GNU General Public License as published by the Free
   Software Foundation; either version 2, or (at your option) any later
   version.

   Bash is distributed in the hope that it will be useful, but WITHOUT ANY
   WARRANTY; without even the implied warranty of MERCHANTABILITY or
   FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
   for more details.

   You should have received a copy of the GNU General Public License along
   with Bash; see the file COPYING.  If not, write to the Free Software
   Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307, USA */
   
#include <config.h>

#if !defined (HAVE_STRERROR)

#include <bashtypes.h>
#ifndef _MINIX
#  include <sys/param.h>
#endif

#if defined (HAVE_UNISTD_H)
#  include <unistd.h>
#endif

#include <stdio.h>
#include <errno.h>

#include <shell.h>

#if !defined (errno)
extern int errno;
#endif /* !errno */

/* Return a string corresponding to the error number E.  From
   the ANSI C spec. */
#if defined (strerror)
#  undef strerror
#endif

static char *errbase = "Unknown system error ";

char *
strerror (e)
     int e;
{
  static char emsg[40];
#if defined (HAVE_SYS_ERRLIST)
  extern int sys_nerr;
  extern char *sys_errlist[];

  if (e > 0 && e < sys_nerr)
    return (sys_errlist[e]);
  else
#endif /* HAVE_SYS_ERRLIST */
    {
      char *z;

      z = itos (e);
      strcpy (emsg, errbase);
      strcat (emsg, z);
      free (z);
      return (&emsg[0]);
    }
}
#endif /* HAVE_STRERROR */
