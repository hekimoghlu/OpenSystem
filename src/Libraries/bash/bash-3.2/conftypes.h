/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 2, 2024.
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

#if !defined (_CONFTYPES_H_)
#define _CONFTYPES_H_

/* Placeholder for future modifications if cross-compiling or building a
   `fat' binary, e.g. on Apple Rhapsody.  These values are used in multiple
   files, so they appear here. */
#if !defined (RHAPSODY) && !defined (MACOSX)
#  define HOSTTYPE	CONF_HOSTTYPE
#  define OSTYPE	CONF_OSTYPE
#  define MACHTYPE	CONF_MACHTYPE
#else /* RHAPSODY */
#  if __ppc64__
#    define HOSTTYPE "ppc64"
#  elif __ppc__
#    define HOSTTYPE "powerpc"
#  elif __x86_64__
#    define HOSTTYPE "x86_64"
#  elif defined(__i386__)
#    define HOSTTYPE "i386"
#  elif defined(__arm__)
#    define HOSTTYPE "arm"
#  else
#    define HOSTTYPE CONF_HOSTTYPE
#  endif

#ifdef CONF_OSTYPE
#  define OSTYPE CONF_OSTYPE
#else
#include "ostype.h"
#endif
#  define VENDOR CONF_VENDOR

#  define MACHTYPE HOSTTYPE "-" VENDOR "-" OSTYPE
#endif /* RHAPSODY */

#ifndef HOSTTYPE
#  define HOSTTYPE "unknown"
#endif

#ifndef OSTYPE
#  define OSTYPE "unknown"
#endif

#ifndef MACHTYPE
#  define MACHTYPE "unknown"
#endif

#endif /* _CONFTYPES_H_ */
