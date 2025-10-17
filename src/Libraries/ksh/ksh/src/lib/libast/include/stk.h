/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 3, 2024.
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
 * David Korn
 * AT&T Research
 *
 * Interface definitions for a stack-like storage library
 *
 */

#ifndef _STK_H
#define _STK_H

#include <sfio.h>

#define _Stk_data	_Stak_data

#define stkstd		(&_Stk_data)

#define	Stk_t		Sfio_t

#define STK_SMALL	1		/* small stkopen stack		*/
#define STK_NULL	2		/* return NULL on overflow	*/

#define	stkptr(sp,n)	((char*)((sp)->_data)+(n))
#define stktop(sp)	((char*)(sp)->_next)
#define	stktell(sp)	((sp)->_next-(sp)->_data)
#define stkseek(sp,n)	((n)==0?(char*)((sp)->_next=(sp)->_data):_stkseek(sp,n))

#if _BLD_ast && defined(__EXPORT__)
#define extern		extern __EXPORT__
#endif
#if !_BLD_ast && defined(__IMPORT__)
#define extern		extern __IMPORT__
#endif

extern Sfio_t		_Stk_data;

#undef	extern

#if _BLD_ast && defined(__EXPORT__)
#define extern		__EXPORT__
#endif

extern Stk_t*		stkopen(int);
extern Stk_t*		stkinstall(Stk_t*, char*(*)(int));
extern int		stkclose(Stk_t*);
extern int		stklink(Stk_t*);
extern char*		stkalloc(Stk_t*, size_t);
extern char*		stkcopy(Stk_t*, const char*);
extern char*		stkset(Stk_t*, char*, size_t);
extern char*		_stkseek(Stk_t*, ssize_t);
extern char*		stkfreeze(Stk_t*, size_t);
extern int		stkon(Stk_t*, char*);

#undef	extern

#endif
