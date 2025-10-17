/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 5, 2022.
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
static char sccsid[] = "@(#)closedir.c	8.1 (Berkeley) 6/10/93";
#endif /* LIBC_SCCS and not lint */
#include <sys/cdefs.h>
__FBSDID("$FreeBSD$");

#include "namespace.h"
#include <sys/types.h>
#include <dirent.h>
#include <pthread.h>
#include <stdlib.h>
#include <unistd.h>
#include "un-namespace.h"

#include "libc_private.h"
#include "telldir.h"

/*
 * close a directory.
 */
static int
_fdclosedir(DIR *dirp)
{
	int fd;

	if (__isthreaded)
		_pthread_mutex_lock(&dirp->dd_lock);
	fd = dirp->dd_fd;
	dirp->dd_fd = -1;
	dirp->dd_loc = 0;
	free((void *)dirp->dd_buf);
	_reclaim_telldir(dirp);
	if (__isthreaded) {
		_pthread_mutex_unlock(&dirp->dd_lock);
		_pthread_mutex_destroy(&dirp->dd_lock);
	}
	free((void *)dirp);
	return (fd);
}

int
closedir(DIR *dirp)
{

	return (_close(_fdclosedir(dirp)));
}
