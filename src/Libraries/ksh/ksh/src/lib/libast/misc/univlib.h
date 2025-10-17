/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 9, 2023.
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
 * universe support
 *
 * symbolic link external representation has trailing '\0' and $(...) style
 * conditionals where $(...) corresponds to a kernel object (i.e., probably
 * not environ)
 *
 * universe symlink conditionals use $(UNIVERSE)
 */

#ifndef _UNIVLIB_H
#define _UNIVLIB_H

#if defined(__STDPP__directive) && defined(__STDPP__hide)
__STDPP__directive pragma pp:hide getuniverse readlink setuniverse symlink universe
#else
#define getuniverse	______getuniverse
#define readlink	______readlink
#define setuniverse	______setuniverse
#define symlink		______symlink
#define universe	______universe
#endif

#include <ast.h>
#include <ls.h>
#include <errno.h>

#define UNIV_SIZE	9

#if _cmd_universe && _sys_universe
#include <sys/universe.h>
#endif

#if defined(__STDPP__directive) && defined(__STDPP__hide)
__STDPP__directive pragma pp:nohide getuniverse readlink setuniverse symlink universe
#else
#undef	getuniverse
#undef	readlink
#undef	setuniverse
#undef	symlink
#undef	universe
#endif

#if _cmd_universe
#ifdef NUMUNIV
#define UNIV_MAX	NUMUNIV
#else
#define UNIV_MAX	univ_max
extern char*		univ_name[];
extern int		univ_max;
#endif

extern char		univ_cond[];
extern int		univ_size;

#else

extern char		univ_env[];

#endif

extern int		getuniverse(char*);
extern int		readlink(const char*, char*, int);
extern int		setuniverse(int);
extern int		symlink(const char*, const char*);
extern int		universe(int);

#endif
