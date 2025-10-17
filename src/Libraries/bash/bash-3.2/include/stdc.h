/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 1, 2025.
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
/* Copyright (C) 1993 Free Software Foundation, Inc.

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

#if !defined (_STDC_H_)
#define _STDC_H_

/* Adapted from BSD /usr/include/sys/cdefs.h. */

/* A function can be defined using prototypes and compile on both ANSI C
   and traditional C compilers with something like this:
	extern char *func __P((char *, char *, int)); */

#if !defined (__P)
#  if defined (__STDC__) || defined (__GNUC__) || defined (__cplusplus) || defined (PROTOTYPES)
#    define __P(protos) protos
#  else 
#    define __P(protos) ()
#  endif
#endif

#if defined (HAVE_STRINGIZE)
#  define __STRING(x) #x
#else
#  define __STRING(x) "x"
#endif

#if !defined (__STDC__)

#if defined (__GNUC__)		/* gcc with -traditional */
#  if !defined (signed)
#    define signed __signed
#  endif
#  if !defined (volatile)
#    define volatile __volatile
#  endif
#else /* !__GNUC__ */
#  if !defined (inline)
#    define inline
#  endif
#  if !defined (signed)
#    define signed
#  endif
#  if !defined (volatile)
#    define volatile
#  endif
#endif /* !__GNUC__ */

#endif /* !__STDC__ */

#ifndef __attribute__
#  if __GNUC__ < 2 || (__GNUC__ == 2 && __GNUC_MINOR__ < 8)
#    define __attribute__(x)
#  endif
#endif

/* For those situations when gcc handles inlining a particular function but
   other compilers complain. */
#ifdef __GNUC__
#  define INLINE inline
#else
#  define INLINE
#endif

#if defined (PREFER_STDARG)
#  define SH_VA_START(va, arg)  va_start(va, arg)
#else
#  define SH_VA_START(va, arg)  va_start(va)
#endif

#endif /* !_STDC_H_ */
