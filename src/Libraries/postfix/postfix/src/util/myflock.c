/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 12, 2025.
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
/* System library. */

#include "sys_defs.h"
#include <errno.h>
#include <unistd.h>

#ifdef HAS_FCNTL_LOCK
#include <fcntl.h>
#include <string.h>
#endif

#ifdef HAS_FLOCK_LOCK
#include <sys/file.h>
#endif

/* Utility library. */

#include "msg.h"
#include "myflock.h"

/* myflock - lock/unlock entire open file */

int     myflock(int fd, int lock_style, int operation)
{
    int     status;

    /*
     * Sanity check.
     */
    if ((operation & (MYFLOCK_OP_BITS)) != operation)
	msg_panic("myflock: improper operation type: 0x%x", operation);

    switch (lock_style) {

	/*
	 * flock() does exactly what we need. Too bad it is not standard.
	 */
#ifdef HAS_FLOCK_LOCK
    case MYFLOCK_STYLE_FLOCK:
	{
	    static int lock_ops[] = {
		LOCK_UN, LOCK_SH, LOCK_EX, -1,
		-1, LOCK_SH | LOCK_NB, LOCK_EX | LOCK_NB, -1
	    };

	    while ((status = flock(fd, lock_ops[operation])) < 0
		   && errno == EINTR)
		sleep(1);
	    break;
	}
#endif

	/*
	 * fcntl() is standard and does more than we need, but we can handle
	 * it.
	 */
#ifdef HAS_FCNTL_LOCK
    case MYFLOCK_STYLE_FCNTL:
	{
	    struct flock lock;
	    int     request;
	    static int lock_ops[] = {
		F_UNLCK, F_RDLCK, F_WRLCK
	    };

	    memset((void *) &lock, 0, sizeof(lock));
	    lock.l_type = lock_ops[operation & ~MYFLOCK_OP_NOWAIT];
	    request = (operation & MYFLOCK_OP_NOWAIT) ? F_SETLK : F_SETLKW;
	    while ((status = fcntl(fd, request, &lock)) < 0
		   && errno == EINTR)
		sleep(1);
	    break;
	}
#endif
    default:
	msg_panic("myflock: unsupported lock style: 0x%x", lock_style);
    }

    /*
     * Return a consistent result. Some systems return EACCES when a lock is
     * taken by someone else, and that would complicate error processing.
     */
    if (status < 0 && (operation & MYFLOCK_OP_NOWAIT) != 0)
	if (errno == EAGAIN || errno == EWOULDBLOCK || errno == EACCES)
	    errno = EAGAIN;

    return (status);
}
