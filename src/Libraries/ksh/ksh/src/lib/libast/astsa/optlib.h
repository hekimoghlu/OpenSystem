/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 10, 2023.
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
/*
 * Glenn Fowler
 * AT&T Research
 *
 * command line option parser and usage formatter private definitions
 */

#ifndef _OPTLIB_H
#define _OPTLIB_H		1

#include <ast.h>
#include <cdt.h>

#define OPT_cache		0x01
#define OPT_functions		0x02
#define OPT_ignore		0x04
#define OPT_long		0x08
#define OPT_old			0x10
#define OPT_plus		0x20
#define OPT_proprietary		0x40

#define OPT_cache_flag		0x01
#define OPT_cache_invert	0x02
#define OPT_cache_numeric	0x04
#define OPT_cache_optional	0x08
#define OPT_cache_string	0x10

#define OPT_CACHE		128
#define OPT_FLAGS		"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

struct Optdisc_s;

typedef struct Optpass_s
{
	char*			opts;
	char*			oopts;
	char*			catalog;
	unsigned char		version;
	unsigned char		prefix;
	unsigned char		flags;
	unsigned char		section;
} Optpass_t;

typedef struct Optcache_s
{
	struct Optcache_s*	next;
	Optpass_t		pass;
	int			caching;
	unsigned char		flags[sizeof(OPT_FLAGS)];
} Optcache_t;

typedef struct Optstate_s
{
	Sfio_t*		mp;		/* opt_info.msg string stream	*/
	Sfio_t*		vp;		/* translation string stream	*/
	Sfio_t*		xp;		/* translation string stream	*/
	Sfio_t*		cp;		/* compatibility string stream	*/
	Optpass_t	pass[8];	/* optjoin() list		*/
	char*		argv[2];	/* initial argv copy		*/
	char*		strv[3];	/* optstr() argv		*/
	char*		str;		/* optstr() string		*/
	Sfio_t*		strp;		/* optstr() stream		*/
	int		force;		/* force this style		*/
	int		pindex;		/* prev index for backup	*/
	int		poffset;	/* prev offset for backup	*/
	int		npass;		/* # optjoin() passes		*/
	int		join;		/* optjoin() pass #		*/
	int		plus;		/* + ok				*/
	int		style;		/* default opthelp() style	*/
	int		width;		/* format line width		*/
	int		flags;		/* display flags		*/
	int		emphasis;	/* ansi term emphasis ok	*/
	Dtdisc_t	msgdisc;	/* msgdict discipline		*/
	Dt_t*		msgdict;	/* default ast.id catalog msgs	*/
	Optcache_t*	cache;		/* OPT_cache cache		*/
} Optstate_t;

#define _OPT_PRIVATE_ \
	char            pad[2*sizeof(void*)]; \
	Optstate_t*	state;

#include <error.h>

#endif
