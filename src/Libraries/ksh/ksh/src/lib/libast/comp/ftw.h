/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 21, 2023.
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
 * ftw,nftw over ftwalk
 */

#ifndef _FTW_H
#define _FTW_H

#define FTW		FTWALK
#include <ftwalk.h>
#undef			FTW

#define FTW_SLN		(FTW_SL|FTW_NR)

#define FTW_PHYS	(FTW_PHYSICAL)
#define FTW_CHDIR	(FTW_DOT)
#define FTW_DEPTH	(FTW_POST)
#define FTW_OPEN	(0)

struct FTW
{
	int		quit;
	int		base;
	int		level;
};

#define FTW_SKD		FTW_SKIP
#define FTW_PRUNE	FTW_SKIP

#if _BLD_ast && defined(__EXPORT__)
#define extern		__EXPORT__
#endif

extern int	ftw(const char*, int(*)(const char*, const struct stat*, int), int);
extern int	nftw(const char*, int(*)(const char*, const struct stat*, int, struct FTW*), int, int);

#undef	extern

#endif
