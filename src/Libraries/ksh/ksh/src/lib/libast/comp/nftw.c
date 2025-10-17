/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 25, 2021.
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
 * nftw implementation
 */

#include <ast.h>
#include <ftw.h>

static int	nftw_flags;
static int	(*nftw_userf)(const char*, const struct stat*, int, struct FTW*);

static int
nftw_user(Ftw_t* ftw)
{
	register int	n = ftw->info;
	struct FTW	nftw;
	struct stat	st;

	if (n & (FTW_C|FTW_NX))
		n = FTW_DNR;
	else if ((n & FTW_SL) && (!(nftw_flags & FTW_PHYSICAL) || stat(ftw->path, &st)))
		n = FTW_SLN;
	nftw.base = ftw->pathlen - ftw->namelen;
	nftw.level = ftw->level;
	nftw.quit = 0;
	n = (*nftw_userf)(ftw->path, &ftw->statb, n, &nftw);
	ftw->status = nftw.quit;
	return n;
}

int
nftw(const char* path, int(*userf)(const char*, const struct stat*, int, struct FTW*), int depth, int flags)
{
	NoP(depth);
	nftw_userf = userf;
	if (flags & FTW_CHDIR) flags &= ~FTW_DOT;
	else flags |= FTW_DOT;
	nftw_flags = flags;
	return ftwalk(path, nftw_user, flags, NiL);
}
