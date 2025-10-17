/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 30, 2025.
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

#if !defined (HAVE_STRCASECMP)

#include <stdc.h>
#include <bashansi.h>
#include <chartypes.h>

/* Compare at most COUNT characters from string1 to string2.  Case
   doesn't matter. */
int
strncasecmp (string1, string2, count)
     const char *string1;
     const char *string2;
     int count;
{
  register const char *s1;
  register const char *s2;
  register int r;

  if (count <= 0 || (string1 == string2))
    return 0;

  s1 = string1;
  s2 = string2;
  do
    {
      if ((r = TOLOWER ((unsigned char) *s1) - TOLOWER ((unsigned char) *s2)) != 0)
	return r;
      if (*s1++ == '\0')
	break;
      s2++;
    }
  while (--count != 0);

  return (0);
}

/* strcmp (), but caseless. */
int
strcasecmp (string1, string2)
     const char *string1;
     const char *string2;
{
  register const char *s1;
  register const char *s2;
  register int r;

  s1 = string1;
  s2 = string2;

  if (s1 == s2)
    return (0);

  while ((r = TOLOWER ((unsigned char)*s1) - TOLOWER ((unsigned char)*s2)) == 0)
    {
      if (*s1++ == '\0')
	return 0;
      s2++;
    }

  return (r);
}
#endif /* !HAVE_STRCASECMP */
