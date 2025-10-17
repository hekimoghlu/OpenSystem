/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 8, 2023.
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
#ifndef _STDLIB
#define _STDLIB

#include <tcl.h>

extern void		abort _ANSI_ARGS_((void));
extern double		atof _ANSI_ARGS_((CONST char *string));
extern int		atoi _ANSI_ARGS_((CONST char *string));
extern long		atol _ANSI_ARGS_((CONST char *string));
extern char *		calloc _ANSI_ARGS_((unsigned int numElements,
			    unsigned int size));
extern void		exit _ANSI_ARGS_((int status));
extern int		free _ANSI_ARGS_((char *blockPtr));
extern char *		getenv _ANSI_ARGS_((CONST char *name));
extern char *		malloc _ANSI_ARGS_((unsigned int numBytes));
extern void		qsort _ANSI_ARGS_((VOID *base, int n, int size,
			    int (*compar)(CONST VOID *element1, CONST VOID
			    *element2)));
extern char *		realloc _ANSI_ARGS_((char *ptr, unsigned int numBytes));
extern double		strtod _ANSI_ARGS_((CONST char *string, char **endPtr));
extern long		strtol _ANSI_ARGS_((CONST char *string, char **endPtr,
			    int base));
extern unsigned long	strtoul _ANSI_ARGS_((CONST char *string,
			    char **endPtr, int base));

#endif /* _STDLIB */
