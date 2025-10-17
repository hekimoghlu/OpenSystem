/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 27, 2022.
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

#if !defined (_XMALLOC_H_)
#define _XMALLOC_H_

#include "stdc.h"
#include "bashansi.h"

/* Generic pointer type. */
#ifndef PTR_T

#if defined (__STDC__)
#  define PTR_T	void *
#else
#  define PTR_T char *
#endif

#endif /* PTR_T */

/* Allocation functions in xmalloc.c */
extern PTR_T xmalloc __P((size_t));
extern PTR_T xrealloc __P((void *, size_t));
extern void xfree __P((void *));

#if defined(USING_BASH_MALLOC) && !defined (DISABLE_MALLOC_WRAPPERS)
extern PTR_T sh_xmalloc __P((size_t, const char *, int));
extern PTR_T sh_xrealloc __P((void *, size_t, const char *, int));
extern void sh_xfree __P((void *, const char *, int));

#define xmalloc(x)	sh_xmalloc((x), __FILE__, __LINE__)
#define xrealloc(x, n)	sh_xrealloc((x), (n), __FILE__, __LINE__)
#define xfree(x)	sh_xfree((x), __FILE__, __LINE__)

#ifdef free
#undef free
#endif
#define free(x)		sh_xfree((x), __FILE__, __LINE__)
#endif	/* USING_BASH_MALLOC */

#endif	/* _XMALLOC_H_ */
