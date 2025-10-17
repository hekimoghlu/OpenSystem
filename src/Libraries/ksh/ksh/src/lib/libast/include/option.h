/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 20, 2025.
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
 * command line option parse interface
 */

#ifndef _OPTION_H
#define _OPTION_H

#include <ast.h>

#define OPT_VERSION	20070319L

#define OPT_USER	(1L<<16)	/* first user flag bit		*/

struct Opt_s;
struct Optdisc_s;

typedef int (*Optinfo_f)(struct Opt_s*, Sfio_t*, const char*, struct Optdisc_s*);

typedef struct Optdisc_s
{
	unsigned long	version;	/* OPT_VERSION			*/
	unsigned long	flags;		/* OPT_* flags			*/
	char*		catalog;	/* error catalog id		*/
	Optinfo_f	infof;		/* runtime info function	*/
} Optdisc_t;

/* NOTE: Opt_t member order fixed by a previous binary release */

#ifndef _OPT_PRIVATE_
#define _OPT_PRIVATE_	\
	char		pad[3*sizeof(void*)];
#endif

typedef struct Opt_s
{
	int		again;		/* see optjoin()		*/
	char*		arg;		/* {:,#} string argument	*/
	char**		argv;		/* most recent argv		*/
	int		index;		/* argv index			*/
	char*		msg;		/* error/usage message buffer	*/
	long		num;		/* OBSOLETE -- use number	*/
	int		offset;		/* char offset in argv[index]	*/
	char		option[8];	/* current flag {-,+} + option  */
	char		name[64];	/* current long name or flag	*/
	Optdisc_t*	disc;		/* user discipline		*/
	intmax_t	number;		/* # numeric argument		*/
	unsigned char	assignment;	/* option arg assigment op	*/
	unsigned char	pads[sizeof(void*)-1];
	_OPT_PRIVATE_
} Opt_t;

#if _BLD_ast && defined(__EXPORT__)
#define extern		extern __EXPORT__
#endif
#if !_BLD_ast && defined(__IMPORT__)
#define extern		extern __IMPORT__
#endif

extern Opt_t*		_opt_infop_;

#define opt_info	(*_opt_infop_)

#undef	extern

#define optinit(d,f)	(memset(d,0,sizeof(*(d))),(d)->version=OPT_VERSION,(d)->infof=(f),opt_info.disc=(d))

#if _BLD_ast && defined(__EXPORT__)
#define extern		__EXPORT__
#endif

extern int		optget(char**, const char*);
extern int		optjoin(char**, ...);
extern char*		opthelp(const char*, const char*);
extern char*		optusage(const char*);
extern int		optstr(const char*, const char*);
extern int		optesc(Sfio_t*, const char*, int);
extern Opt_t*		optctx(Opt_t*, Opt_t*);

#undef	extern

#endif
