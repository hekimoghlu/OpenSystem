/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 27, 2025.
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

#include <stdio.h>
#include <stdlib.h>
#ifdef HAVE_STDBOOL_H
# include <stdbool.h>
#else
# include "compat/stdbool.h"
#endif
#include <errno.h>
#include <fcntl.h>
#include <time.h>
#include <unistd.h>

#include "sudo_compat.h"
#include "sudo_debug.h"
#include "sudo_iolog.h"
#include "sudo_util.h"

static unsigned char const gzip_magic[2] = {0x1f, 0x8b};

/*
 * Open the specified I/O log file and store in iol.
 * Stores the open file handle which has the close-on-exec flag set.
 */
bool
iolog_open(struct iolog_file *iol, int dfd, int iofd, const char *mode)
{
    int flags;
    const char *file;
    unsigned char magic[2];
    const uid_t iolog_uid = iolog_get_uid();
    const gid_t iolog_gid = iolog_get_gid();
    debug_decl(iolog_open, SUDO_DEBUG_UTIL);

    if (mode[0] == 'r') {
	flags = mode[1] == '+' ? O_RDWR : O_RDONLY;
    } else if (mode[0] == 'w') {
	flags = O_CREAT|O_TRUNC;
	flags |= mode[1] == '+' ? O_RDWR : O_WRONLY;
    } else {
	sudo_debug_printf(SUDO_DEBUG_ERROR,
	    "%s: invalid I/O mode %s", __func__, mode);
	debug_return_bool(false);
    }
    if ((file = iolog_fd_to_name(iofd)) == NULL) {
	sudo_debug_printf(SUDO_DEBUG_ERROR,
	    "%s: invalid iofd %d", __func__, iofd);
	debug_return_bool(false);
    }

    iol->writable = false;
    iol->compressed = false;
    if (iol->enabled) {
	int fd = iolog_openat(dfd, file, flags);
	if (fd != -1) {
	    if (*mode == 'w') {
		if (fchown(fd, iolog_uid, iolog_gid) != 0) {
		    sudo_debug_printf(SUDO_DEBUG_ERROR|SUDO_DEBUG_ERRNO,
			"%s: unable to fchown %d:%d %s", __func__,
			(int)iolog_uid, (int)iolog_gid, file);
		}
		iol->compressed = iolog_get_compress();
	    } else {
		/* check for gzip magic number */
		if (pread(fd, magic, sizeof(magic), 0) == ssizeof(magic)) {
		    if (magic[0] == gzip_magic[0] && magic[1] == gzip_magic[1])
			iol->compressed = true;
		}
	    }
	    if (fcntl(fd, F_SETFD, FD_CLOEXEC) != -1) {
#ifdef HAVE_ZLIB_H
		if (iol->compressed)
		    iol->fd.g = gzdopen(fd, mode);
		else
#endif
		    iol->fd.f = fdopen(fd, mode);
	    }
	    if (iol->fd.v != NULL) {
		switch ((flags & O_ACCMODE)) {
		case O_WRONLY:
		case O_RDWR:
		    iol->writable = true;
		    break;
		}
	    } else {
		int save_errno = errno;
		close(fd);
		errno = save_errno;
		fd = -1;
	    }
	}
	if (fd == -1) {
	    iol->enabled = false;
	    debug_return_bool(false);
	}
    } else {
	if (*mode == 'w') {
	    /* Remove old log file in case we recycled sequence numbers. */
	    (void)unlinkat(dfd, file, 0);
	}
    }
    debug_return_bool(true);
}
