/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 16, 2023.
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

#include <ast.h>

#if !defined(getgroups) && defined(_lib_getgroups)

NoN(getgroups)

#else

#include <error.h>

#if defined(getgroups)
#undef	getgroups
#define	ast_getgroups	_ast_getgroups
#define botched		1
extern int		getgroups(int, int*);
#else
#define ast_getgroups	getgroups
#endif

#if defined(__EXPORT__)
#define extern	__EXPORT__
#endif

extern int
ast_getgroups(int len, gid_t* set)
{
#if botched
#if NGROUPS_MAX < 1
#undef	NGROUPS_MAX
#define NGROUPS_MAX	1
#endif
	register int	i;
	int		big[NGROUPS_MAX];
#else
#undef	NGROUPS_MAX
#define NGROUPS_MAX	1
#endif
	if (!len) return(NGROUPS_MAX);
	if (len < 0 || !set)
	{
		errno = EINVAL;
		return(-1);
	}
#if botched
	len = getgroups(len > NGROUPS_MAX ? NGROUPS_MAX : len, big);
	for (i = 0; i < len; i++)
		set[i] = big[i];
	return(len);
#else
	*set = getgid();
	return(1);
#endif
}

#endif
