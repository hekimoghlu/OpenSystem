/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 12, 2022.
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
#if defined(LIBC_SCCS) && !defined(lint)
static char sccsid[] = "@(#)rewinddir.c	8.1 (Berkeley) 6/8/93";
#endif /* LIBC_SCCS and not lint */
#include <sys/cdefs.h>
__FBSDID("$FreeBSD$");

#include "namespace.h"
#include <sys/types.h>
#include <dirent.h>
#include <pthread.h>
#include <unistd.h>
#include "un-namespace.h"

#include "libc_private.h"
#include "telldir.h"

void
rewinddir(DIR *dirp)
{

	if (__isthreaded)
		_pthread_mutex_lock(&dirp->dd_lock);
	dirp->dd_flags &= ~(__DTF_SKIPREAD | __DTF_ATEND); /* current contents are invalid */
	if (dirp->dd_flags & __DTF_READALL)
		_filldir(dirp, false);
	else {
		(void) lseek(dirp->dd_fd, 0, SEEK_SET);
#if __DARWIN_64_BIT_INO_T
		dirp->dd_td->seekoff = 0;
#else /* !__DARWIN_64_BIT_INO_T */
		dirp->dd_seek = 0;
#endif /* __DARWIN_64_BIT_INO_T */
	}
	dirp->dd_loc = 0;
	_reclaim_telldir(dirp);
	if (__isthreaded)
		_pthread_mutex_unlock(&dirp->dd_lock);
}
