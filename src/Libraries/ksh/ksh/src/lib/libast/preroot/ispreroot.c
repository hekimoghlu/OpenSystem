/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 6, 2022.
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
 * AT&T Bell Laboratories
 * return 1 if dir [any dir] is the preroot
 */

#include <ast.h>
#include <preroot.h>

#if FS_PREROOT

#include <ls.h>

/*
 * return 1 if files a and b are the same under preroot
 *
 * NOTE: the kernel disables preroot for set-uid processes
 */

static int
same(const char* a, const char* b)
{
	int		i;
	int		euid;
	int		ruid;

	struct stat	ast;
	struct stat	bst;

	if ((ruid = getuid()) != (euid = geteuid())) setuid(ruid);
	i = !stat(a, &ast) && !stat(b, &bst) && ast.st_dev == bst.st_dev && ast.st_ino == bst.st_ino;
	if (ruid != euid) setuid(euid);
	return(i);
}

int
ispreroot(const char* dir)
{
	static int	prerooted = -1;

	if (dir) return(same("/", dir));
	if (prerooted < 0) prerooted = !same("/", PR_REAL);
	return(prerooted);
}

#else

NoN(ispreroot)

#endif
