/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 29, 2022.
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
#pragma prototyped
/*
 * posix wordexp interface definitions
 */

#ifndef _WORDEXP_H
#define _WORDEXP_H

#include <ast_common.h>

#define WRDE_APPEND	01
#define WRDE_DOOFFS	02
#define WRDE_NOCMD	04
#define WRDE_NOSYS	0100
#define WRDE_REUSE	010
#define WRDE_SHOWERR	020
#define WRDE_UNDEF	040

#define WRDE_BADCHAR	1
#define WRDE_BADVAL	2
#define WRDE_CMDSUB	3
#define WRDE_NOSPACE	4
#define WRDE_SYNTAX	5
#define WRDE_NOSHELL	6

typedef struct _wdarg
{
	size_t	we_wordc;
	char	**we_wordv;
	size_t	we_offs;
} wordexp_t;

#if _BLD_ast && defined(__EXPORT__)
#define extern		__EXPORT__
#endif

extern int wordexp(const char*, wordexp_t*, int);
extern int wordfree(wordexp_t*);

#undef	extern

#endif /* _WORDEXP_H */
