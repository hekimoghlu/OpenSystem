/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 1, 2022.
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
 * This is an open source non-commercial project. Dear PVS-Studio, please check it.
 * PVS-Studio Static Code Analyzer for C, C++ and C#: http://www.viva64.com
 */

#include <config.h>

#include <stdlib.h>
#include <string.h>
#ifdef HAVE_STDBOOL_H
# include <stdbool.h>
#else
# include "compat/stdbool.h"
#endif
#include <errno.h>
#include <fcntl.h>
#include <unistd.h>

#include "sudo_compat.h"
#include "sudo_util.h"
#include "sudo_debug.h"

bool
sudo_lock_file_v1(int fd, int type)
{
    return sudo_lock_region_v1(fd, type, 0);
}

/*
 * Lock/unlock all or part of a file.
 */
#ifdef HAVE_LOCKF
bool
sudo_lock_region_v1(int fd, int type, off_t len)
{
    int op, rc;
    off_t oldpos = -1;
    debug_decl(sudo_lock_region, SUDO_DEBUG_UTIL);

    switch (type) {
	case SUDO_LOCK:
	    sudo_debug_printf(SUDO_DEBUG_INFO, "%s: lock %d:%lld",
		__func__, fd, (long long)len);
	    op = F_LOCK;
	    break;
	case SUDO_TLOCK:
	    sudo_debug_printf(SUDO_DEBUG_INFO, "%s: tlock %d:%lld",
		__func__, fd, (long long)len);
	    op = F_TLOCK;
	    break;
	case SUDO_UNLOCK:
	    sudo_debug_printf(SUDO_DEBUG_INFO, "%s: unlock %d:%lld",
		__func__, fd, (long long)len);
	    op = F_ULOCK;
	    /* Must seek to start of file to unlock the entire thing. */
	    if (len == 0 && (oldpos = lseek(fd, 0, SEEK_CUR)) != -1) {
		if (lseek(fd, 0, SEEK_SET) == -1) {
		    sudo_debug_printf(
			SUDO_DEBUG_ERROR|SUDO_DEBUG_LINENO|SUDO_DEBUG_ERRNO,
			"unable to seek to beginning");
		}
	    }
	    break;
	default:
	    sudo_debug_printf(SUDO_DEBUG_INFO, "%s: bad lock type %d",
		__func__, type);
	    errno = EINVAL;
	    debug_return_bool(false);
    }
    rc = lockf(fd, op, len);
    if (oldpos != -1) {
	if (lseek(fd, oldpos, SEEK_SET) == -1) {
	    sudo_debug_printf(
		SUDO_DEBUG_ERROR|SUDO_DEBUG_LINENO|SUDO_DEBUG_ERRNO,
		"unable to restore offset");
	}
    }
    debug_return_bool(rc == 0);
}
#else
bool
sudo_lock_region_v1(int fd, int type, off_t len)
{
    struct flock lock;
    int func;
    debug_decl(sudo_lock_file, SUDO_DEBUG_UTIL);

    switch (type) {
	case SUDO_LOCK:
	    sudo_debug_printf(SUDO_DEBUG_INFO, "%s: lock %d:%lld",
		__func__, fd, (long long)len);
	    lock.l_type = F_WRLCK;
	    func = F_SETLKW;
	    break;
	case SUDO_TLOCK:
	    sudo_debug_printf(SUDO_DEBUG_INFO, "%s: tlock %d:%lld",
		__func__, fd, (long long)len);
	    lock.l_type = F_WRLCK;
	    func = F_SETLK;
	    break;
	case SUDO_UNLOCK:
	    sudo_debug_printf(SUDO_DEBUG_INFO, "%s: unlock %d:%lld",
		__func__, fd, (long long)len);
	    lock.l_type = F_UNLCK;
	    func = F_SETLK;
	    break;
	default:
	    sudo_debug_printf(SUDO_DEBUG_INFO, "%s: bad lock type %d",
		__func__, type);
	    errno = EINVAL;
	    debug_return_bool(false);
    }
    lock.l_start = 0;
    lock.l_len = len;
    lock.l_pid = 0;
    lock.l_whence = len ? SEEK_CUR : SEEK_SET;

    debug_return_bool(fcntl(fd, func, &lock) == 0);
}
#endif
