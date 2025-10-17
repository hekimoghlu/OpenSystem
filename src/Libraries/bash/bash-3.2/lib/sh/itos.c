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
/* Copyright (C) 1998-2002 Free Software Foundation, Inc.

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
   Foundation, 59 Temple Place, Suite 330, Boston, MA 02111 USA. */

#ifdef HAVE_CONFIG_H
#  include <config.h>
#endif

#if defined (HAVE_UNISTD_H)
#  include <unistd.h>
#endif

#include <bashansi.h>
#include "shell.h"

char *
inttostr (i, buf, len)
     intmax_t i;
     char *buf;
     size_t len;
{
  return (fmtumax (i, 10, buf, len, 0));
}

/* Integer to string conversion.  This conses the string; the
   caller should free it. */
char *
itos (i)
     intmax_t i;
{
  char *p, lbuf[INT_STRLEN_BOUND(intmax_t) + 1];

  p = fmtumax (i, 10, lbuf, sizeof(lbuf), 0);
  return (savestring (p));
}

char *
uinttostr (i, buf, len)
     uintmax_t i;
     char *buf;
     size_t len;
{
  return (fmtumax (i, 10, buf, len, FL_UNSIGNED));
}

/* Integer to string conversion.  This conses the string; the
   caller should free it. */
char *
uitos (i)
     uintmax_t i;
{
  char *p, lbuf[INT_STRLEN_BOUND(uintmax_t) + 1];

  p = fmtumax (i, 10, lbuf, sizeof(lbuf), FL_UNSIGNED);
  return (savestring (p));
}
