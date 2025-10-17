/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 24, 2022.
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

#ifndef _STDHDR_H
#define _STDHDR_H	1

#ifndef _NO_LARGEFILE64_SOURCE
#define _NO_LARGEFILE64_SOURCE	1
#endif
#undef	_LARGEFILE64_SOURCE

#define _ast_fseeko	______fseeko
#define _ast_ftello	______ftello
#include "sfhdr.h"
#undef	_ast_fseeko
#undef	_ast_ftello

#include "stdio.h"

#define SF_MB		010000
#define SF_WC		020000

#if _UWIN

#define STDIO_TRANSFER	1

typedef int (*Fun_f)();

typedef struct Funvec_s
{
	const char*	name;
	Fun_f		vec[2];
} Funvec_t;

extern int	_stdfun(Sfio_t*, Funvec_t*);

#define STDIO_INT(p,n,t,f,a) \
	{ \
		typedef t (*_s_f)f; \
		int		_i; \
		static Funvec_t	_v = { n }; \
		if ((_i = _stdfun(p, &_v)) < 0) \
			return -1; \
		else if (_i > 0) \
			return ((_s_f)_v.vec[_i])a; \
	}

#define STDIO_PTR(p,n,t,f,a) \
	{ \
		typedef t (*_s_f)f; \
		int		_i; \
		static Funvec_t	_v = { n }; \
		if ((_i = _stdfun(p, &_v)) < 0) \
			return 0; \
		else if (_i > 0) \
			return ((_s_f)_v.vec[_i])a; \
	}

#define STDIO_VOID(p,n,t,f,a) \
	{ \
		typedef t (*_s_f)f; \
		int		_i; \
		static Funvec_t	_v = { n }; \
		if ((_i = _stdfun(p, &_v)) < 0) \
			return; \
		else if (_i > 0) \
		{ \
			((_s_f)_v.vec[_i])a; \
			return; \
		} \
	}

#else

#define STDIO_INT(p,n,t,f,a)
#define STDIO_PTR(p,n,t,f,a)
#define STDIO_VOID(p,n,t,f,a)

#endif

#define FWIDE(f,r) \
	do \
	{ \
		if (fwide(f, 0) < 0) \
			return r; \
		f->bits |= SF_WC; \
	} while (0)

#ifdef __EXPORT__
#define extern	__EXPORT__
#endif

extern int		sfdcwide(Sfio_t*);

#endif
