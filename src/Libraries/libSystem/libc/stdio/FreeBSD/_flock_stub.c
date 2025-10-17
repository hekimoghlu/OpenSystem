/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 10, 2022.
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
 * POSIX stdio FILE locking functions. These assume that the locking
 * is only required at FILE structure level, not at file descriptor
 * level too.
 *
 */

#include <sys/cdefs.h>
__FBSDID("$FreeBSD: src/lib/libc/stdio/_flock_stub.c,v 1.16 2008/04/17 22:17:53 jhb Exp $");

#include "namespace.h"
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include "un-namespace.h"

#include "local.h"

/*
 * Weak symbols for externally visible functions in this file:
 */
__weak_reference(_flockfile, flockfile);
__weak_reference(_flockfile_debug_stub, _flockfile_debug);
__weak_reference(_ftrylockfile, ftrylockfile);
__weak_reference(_funlockfile, funlockfile);

void
_flockfile(FILE *fp)
{
	// <rdar://problem/21533199> - preserve errno.
	int save_errno = errno;
	_pthread_mutex_lock(&fp->_fl_mutex);
	errno = save_errno;
}

/*
 * This can be overriden by the threads library if it is linked in.
 */
void
_flockfile_debug_stub(FILE *fp, char *fname, int lineno)
{
	_flockfile(fp);
}

int
_ftrylockfile(FILE *fp)
{
	int	ret = 0;

	// <rdar://problem/21533199> - preserve errno.
	int save_errno = errno;
	if (_pthread_mutex_trylock(&fp->_fl_mutex) != 0)
		ret = -1;
	errno = save_errno;

	return (ret);
}

void 
_funlockfile(FILE *fp)
{
	// <rdar://problem/21533199> - preserve errno.
	int save_errno = errno;
	_pthread_mutex_unlock(&fp->_fl_mutex);
	errno = save_errno;
}
