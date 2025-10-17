/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 17, 2024.
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
/* This file is included at the start of defs.h; this file
 * is an initial attempt to gather in one place some declarations
 * that may need to be tweaked on some systems.
 */

#ifdef __STDC__
#undef KR_headers
#endif

#ifndef KR_headers
#ifndef ANSI_Libraries
#define ANSI_Libraries
#endif
#ifndef ANSI_Prototypes
#define ANSI_Prototypes
#endif
#endif

#ifdef __BORLANDC__
#define MSDOS
#endif

#ifdef __ZTC__	/* Zortech */
#define MSDOS
#endif

#ifdef MSDOS
#define ANSI_Libraries
#define ANSI_Prototypes
#define LONG_CAST (long)
#else
#define LONG_CAST
#endif

#include <stdio.h>

#ifdef ANSI_Libraries
#include <stddef.h>
#include <stdlib.h>
#else
char *calloc(), *malloc(), *realloc();
void *memcpy(), *memset();
#ifndef _SIZE_T
typedef unsigned int size_t;
#endif
#ifndef atol
    long atol();
#endif

#ifdef ANSI_Prototypes
extern double atof(const char *);
extern double strtod(const char*, char**);
#else
extern double atof(), strtod();
#endif
#endif

/* On systems like VMS where fopen might otherwise create
 * multiple versions of intermediate files, you may wish to
 * #define scrub(x) unlink(x)
 */
#ifndef scrub
#define scrub(x) /* do nothing */
#endif

/* On systems that severely limit the total size of statically
 * allocated arrays, you may need to change the following to
 *	extern char **chr_fmt, *escapes, **str_fmt;
 * and to modify sysdep.c appropriately
 */
extern char *chr_fmt[], escapes[], *str_fmt[];

#include <string.h>

#include "ctype.h"

#define Bits_per_Byte 8
#define Table_size (1 << Bits_per_Byte)
