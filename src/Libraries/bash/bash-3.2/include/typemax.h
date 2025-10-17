/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 17, 2023.
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
/* Copyright (C) 2001 Free Software Foundation, Inc.

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

/*
 * NOTE:  This should be included after config.h, limits.h, stdint.h, and
 *	  inttypes.h
 */

#ifndef _SH_TYPEMAX_H
#define _SH_TYPEMAX_H

#ifndef CHAR_BIT
#  define CHAR_BIT 8
#endif

/* Nonzero if the integer type T is signed.  */
#ifndef TYPE_SIGNED
#  define TYPE_SIGNED(t)	(! ((t) 0 < (t) -1))
#endif

#ifndef TYPE_MINIMUM
#  define TYPE_MINIMUM(t) ((t) (TYPE_SIGNED (t) \
				? ~ (t) 0 << (sizeof (t) * CHAR_BIT - 1) \
				: (t) 0))
#endif

#ifndef TYPE_MAXIMUM
#  define TYPE_MAXIMUM(t) ((t) (~ (t) 0 - TYPE_MINIMUM (t)))
#endif

#ifdef HAVE_LONG_LONG
#  ifndef LLONG_MAX
#    define LLONG_MAX   TYPE_MAXIMUM(long long int)
#    define LLONG_MIN	TYPE_MINIMUM(long long int)
#  endif
#  ifndef ULLONG_MAX
#    define ULLONG_MAX  TYPE_MAXIMUM(unsigned long long int)
#  endif
#endif

#ifndef ULONG_MAX
#  define ULONG_MAX	((unsigned long) ~(unsigned long) 0)
#endif

#ifndef LONG_MAX
#  define LONG_MAX	((long int) (ULONG_MAX >> 1))
#  define LONG_MIN	((long int) (-LONG_MAX - 1L))
#endif

#ifndef INT_MAX		/* ouch */
#  define INT_MAX	TYPE_MAXIMUM(int)
#  define INT_MIN	TYPE_MINIMUM(int)
#  define UINT_MAX	((unsigned int) ~(unsigned int)0)
#endif

/* workaround for gcc bug in versions < 2.7 */
#if defined (HAVE_LONG_LONG) && __GNUC__ == 2 && __GNUC_MINOR__ < 7
static const unsigned long long int maxquad = ULLONG_MAX;
#  undef ULLONG_MAX
#  define ULLONG_MAX maxquad
#endif

#endif /* _SH_TYPEMAX_H */
