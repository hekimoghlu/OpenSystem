/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 31, 2025.
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
 * Glenn Fowler
 * AT&T Research
 *
 * character code map interface
 *
 * NOTE: used for mapping between 8-bit character encodings
 *	 ISO character sets are handled by sfio
 */

#ifndef _CHARCODE_H
#define _CHARCODE_H	1

#include <ast_common.h>
#include <ast_ccode.h>

typedef struct Ccmap_s
{
	const char*	name;	/* code set name		*/
	const char*	match;	/* strmatch() pattern		*/
	const char*	desc;	/* code set description		*/
	const char*	canon;	/* canonical name format	*/
	const char*	index;	/* default index		*/
	int		ccode;	/* <ccode.h> code index		*/
	void*		data;	/* map specific data		*/
} Ccmap_t;

#if _BLD_ast && defined(__EXPORT__)
#define extern		__EXPORT__
#endif

extern unsigned char*	_ccmap(int, int);
extern void*		_ccmapcpy(unsigned char*, void*, const void*, size_t);
extern void*		_ccmapstr(unsigned char*, void*, size_t);

extern int		ccmapid(const char*);
extern char*		ccmapname(int);
extern void*		ccnative(void*, const void*, size_t);
extern Ccmap_t*		ccmaplist(Ccmap_t*);

#undef	extern

#define CCOP(i,o)		((i)==(o)?0:(((o)<<8)|(i)))
#define CCIN(x)			((x)&0xFF)
#define CCOUT(x)		(((x)>>8)&0xFF)
#define CCCONVERT(x)		((x)&0xFF00)

#define CCCVT(x)		CCMAP(x,0)
#define CCMAP(i,o)		((i)==(o)?(unsigned char*)0:_ccmap(i,o))
#define CCMAPCHR(m,c)		((m)?(m)[c]:(c))
#define CCMAPCPY(m,t,f,n)	((m)?_ccmapcpy(m,t,f,n):memcpy(t,f,n))
#define CCMAPSTR(m,s,n)		((m)?_ccmapstr(m,s,n):(void*)(s))

#define ccmap(i,o)		CCMAP(i,o)
#define ccmapchr(m,c)		CCMAPCHR(m,c)
#define ccmapcpy(m,t,f,n)	CCMAPCPY(m,t,f,n)
#define ccmapstr(m,s,n)		CCMAPSTR(m,s,n)

#define CCMAPC(c,i,o)		((i)==(o)?(c):CCMAP(i,o)[c])
#define CCMAPM(t,f,n,i,o)	((i)==(o)?memcpy(t,f,n):_ccmapcpy(CCMAP(i,o),t,f,n))
#define CCMAPS(s,n,i,o)		((i)==(o)?(void*)(s):_ccmapstr(CCMAP(i,o),s,n))

#define ccmapc(c,i,o)		CCMAPC(c,i,o)
#define ccmapm(t,f,n,i,o)	CCMAPM(t,f,n,i,o)
#define ccmaps(s,n,i,o)		CCMAPS(s,n,i,o)

#endif
