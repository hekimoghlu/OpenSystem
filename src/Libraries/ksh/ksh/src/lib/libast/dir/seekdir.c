/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 1, 2025.
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
 * seekdir
 *
 * seek on directory stream
 * this is not optimal because there aren't portable
 * semantics for directory seeks
 */

#include "dirlib.h"

#if _dir_ok

NoN(seekdir)

#else

void
seekdir(register DIR* dirp, long loc)
{
	off_t	base;		/* file location of block */
	off_t	offset; 	/* offset within block */

	if (telldir(dirp) != loc)
	{
		lseek(dirp->dd_fd, 0L, SEEK_SET);
		dirp->dd_loc = dirp->dd_size = 0;
		while (telldir(dirp) != loc)
			if (!readdir(dirp))
				break; 	/* "can't happen" */
	}
}

#endif
