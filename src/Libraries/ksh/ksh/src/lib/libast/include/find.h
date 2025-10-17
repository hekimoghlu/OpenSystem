/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 20, 2023.
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
 * fast find interface definitions
 */

#ifndef _FIND_H
#define _FIND_H

#define FIND_VERSION	19980301L

#ifndef FIND_CODES
#define FIND_CODES	"lib/find/codes"
#endif

#define FIND_CODES_ENV	"FINDCODES"

#define FIND_GENERATE	(1<<0)		/* generate new codes		*/
#define FIND_ICASE	(1<<1)		/* ignore case in match		*/
#define FIND_GNU	(1<<2)		/* generate gnu format codes	*/
#define FIND_OLD	(1<<3)		/* generate old format codes	*/
#define FIND_TYPE	(1<<4)		/* generate type with codes	*/
#define FIND_VERIFY	(1<<5)		/* verify the dir hierarchy	*/

#define FIND_USER	(1L<<16)	/* first user flag bit		*/

struct Find_s;
struct Finddisc_s;

typedef int (*Findverify_f)(struct Find_s*, const char*, size_t, struct Finddisc_s*);

typedef struct Finddisc_s
{
	unsigned long	version;	/* interface version		*/
	unsigned long	flags;		/* FIND_* flags			*/
	Error_f		errorf;		/* error function		*/
	Findverify_f	verifyf;	/* dir verify function		*/
	char**		dirs;		/* dir prefixes to search	*/
} Finddisc_t;

typedef struct Find_s
{
	const char*	id;		/* library id string		*/
	unsigned long	stamp;		/* codes time stamp		*/

#ifdef _FIND_PRIVATE_
	_FIND_PRIVATE_
#endif

} Find_t;

#if _BLD_ast && defined(__EXPORT__)
#define extern		__EXPORT__
#endif

extern Find_t*		findopen(const char*, const char*, const char*, Finddisc_t*);
extern char*		findread(Find_t*);
extern int		findwrite(Find_t*, const char*, size_t, const char*);
extern int		findclose(Find_t*);

#undef	extern

#endif
