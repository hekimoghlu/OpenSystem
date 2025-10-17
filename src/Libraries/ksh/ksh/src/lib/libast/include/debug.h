/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 8, 2022.
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
 * common ast debug definitions
 * include after the ast headers
 */

#ifndef _DEBUG_H
#define _DEBUG_H

#include <ast.h>
#include <error.h>

#if !defined(DEBUG) && _BLD_DEBUG
#define DEBUG		_BLD_DEBUG
#endif

#if DEBUG || _BLD_DEBUG

#define debug(x)	x
#define message(x)	do if (error_info.trace < 0) { error x; } while (0)
#define messagef(x)	do if (error_info.trace < 0) { errorf x; } while (0)

#define DEBUG_BEGTIME()		debug_elapsed(1)
#define DEBUG_GETTIME()		debug_elapsed(0)
#define DEBUG_ASSERT(p)		((p) ? 0 : (debug_fatal(__FILE__, __LINE__),0))
#define DEBUG_COUNT(n)		((n) += 1)
#define DEBUG_TALLY(c,n,v)	((c) ? ((n) += (v)) : (n))
#define DEBUG_INCREASE(n)	((n) += 1)
#define DEBUG_DECREASE(n)	((n) -= 1)
#define DEBUG_DECLARE(t,v)	t v
#define DEBUG_SET(n,v)		((n) = (v))
#define DEBUG_PRINT(fd,s,v)	do {char _b[1024];write(fd,_b,sfsprintf(_b,sizeof(_b),s,v));} while(0)
#define DEBUG_WRITE(fd,d,n)	write((fd),(d),(n))
#define DEBUG_TEMP(temp)	(temp) /* debugging stuff that should be removed */
#define DEBUG_BREAK		break
#define DEBUG_CONTINUE		continue
#define DEBUG_GOTO(label)	do { debug_fatal(__FILE__, __LINE__); goto label; } while(0)
#define DEBUG_RETURN(x)		do { debug_fatal(__FILE__, __LINE__); return(x); } while(0)

#else

#define debug(x)
#define message(x)
#define messagef(x)

#define DEBUG_BEGTIME()
#define DEBUG_GETTIME()
#define DEBUG_ASSERT(p)
#define DEBUG_COUNT(n)
#define DEBUG_TALLY(c,n,v)
#define DEBUG_INCREASE(n)
#define DEBUG_DECREASE(n)
#define DEBUG_DECLARE(t,v)
#define DEBUG_SET(n,v)
#define DEBUG_PRINT(fd,s,v)
#define DEBUG_WRITE(fd,d,n)
#define DEBUG_TEMP(x)
#define DEBUG_BREAK		break
#define DEBUG_CONTINUE		continue
#define DEBUG_GOTO(label)	goto label
#define DEBUG_RETURN(x)		return(x)

#endif

#ifndef BREAK
#define BREAK			DEBUG_BREAK
#endif
#ifndef CONTINUE
#define CONTINUE		DEBUG_CONTINUE
#endif
#ifndef GOTO
#define GOTO(label)		DEBUG_GOTO(label)
#endif
#ifndef RETURN
#define RETURN(x)		DEBUG_RETURN(x)
#endif

#if _BLD_ast && defined(__EXPORT__)
#define extern		__EXPORT__
#endif

extern double		debug_elapsed(int);
extern void		debug_fatal(const char*, int);
extern void		systrace(const char*);

#undef	extern

#endif
