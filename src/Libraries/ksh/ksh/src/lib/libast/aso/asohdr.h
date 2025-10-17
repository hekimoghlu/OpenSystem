/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 30, 2024.
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
#ifndef _ASOHDR_H
#define _ASOHDR_H	1

#if _PACKAGE_ast

#include	<ast.h>
#include	<error.h>
#include	<fnv.h>

#else

#include	<errno.h>

#ifndef elementsof
#define elementsof(x)	(sizeof(x)/sizeof(x[0]))
#endif
#ifndef integralof
#define integralof(x)	(((char*)(x))-((char*)0))
#endif
#ifndef FNV_MULT
#define FNV_MULT	0x01000193L
#endif
#ifndef NiL
#define NiL		((void*)0)
#endif
#ifndef NoN 
#if defined(__STDC__) || defined(__STDPP__)
#define NoN(x)		void _STUB_ ## x () {}
#else
#define NoN(x)		void _STUB_/**/x () {}
#endif
#if !defined(_STUB_)
#define _STUB_
#endif
#endif

#endif

#include	"FEATURE/asometh"

#if _UWIN
#undef	_aso_fcntl
#undef	_aso_semaphore
#endif

#include	"aso.h"

#define HASH(p,z)	((integralof(p)*FNV_MULT)%(z))

#endif
