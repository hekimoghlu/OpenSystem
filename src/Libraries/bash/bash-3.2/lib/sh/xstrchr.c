/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 13, 2024.
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
/* Copyright (C) 2002 Free Software Foundation, Inc.

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

#include <config.h>

#ifdef HAVE_STDLIB_H
#  include <stdlib.h>
#endif

#include "bashansi.h"
#include "shmbutil.h"

#undef xstrchr

/* In some locales, the non-first byte of some multibyte characters have
   the same value as some ascii character.  Faced with these strings, a
   legacy strchr() might return the wrong value. */

char *
#if defined (PROTOTYPES)
xstrchr (const char *s, int c)
#else
xstrchr (s, c)
     const char *s;
     int c;
#endif
{
#if HANDLE_MULTIBYTE
  char *pos;
  mbstate_t state;
  size_t strlength, mblength;

  /* The locale encodings with said weird property are BIG5, BIG5-HKSCS,
     GBK, GB18030, SHIFT_JIS, and JOHAB.  They exhibit the problem only
     when c >= 0x30.  We can therefore use the faster bytewise search if
     c <= 0x30. */
  if ((unsigned char)c >= '0' && MB_CUR_MAX > 1)
    {
      pos = (char *)s;
      memset (&state, '\0', sizeof(mbstate_t));
      strlength = strlen (s);

      while (strlength > 0)
	{
	  mblength = mbrlen (pos, strlength, &state);
	  if (mblength == (size_t)-2 || mblength == (size_t)-1 || mblength == (size_t)0)
	    mblength = 1;

	  if (c == (unsigned char)*pos)
	    return pos;

	  strlength -= mblength;
	  pos += mblength;
	}

      return ((char *)NULL);
    }
  else
#endif
  return (strchr (s, c));
}
