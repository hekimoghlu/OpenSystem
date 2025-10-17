/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 19, 2024.
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
#ifndef _SFSTR_H
#define _SFSTR_H	1

#include <ast.h>

typedef struct Sfstr_s
{
	char*		beg;
	char*		nxt;
	char*		end;
} Sfstr_t;

#undef	sfclose
#undef	sfprintf
#undef	sfprints
#undef	sfputc
#undef	sfputr
#undef	sfstrbase
#undef	sfstropen
#undef	sfstrseek
#undef	sfstrset
#undef	sfstrtell
#undef	sfstruse
#undef	sfwrite

extern int	sfclose(Sfio_t*);
extern int	sfprintf(Sfio_t*, const char*, ...);
extern char*	sfprints(const char*, ...);
extern int	sfputc(Sfio_t*, int);
extern int	sfputr(Sfio_t*, const char*, int);
extern char*	sfstrbase(Sfio_t*);
extern Sfio_t*	sfstropen(void);
extern char*	sfstrseek(Sfio_t*, int, int);
extern char*	sfstrset(Sfio_t*, int);
extern int	sfstrtell(Sfio_t*);
extern char*	sfstruse(Sfio_t*);
extern int	sfwrite(Sfio_t*, void*, int);

#endif
