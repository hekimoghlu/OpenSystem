/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 1, 2022.
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

#include <sys/stat.h>
#include <stdio.h>
#include <stdlib.h>
#ifdef HAVE_STDBOOL_H
# include <stdbool.h>
#else
# include "compat/stdbool.h"
#endif
#include <errno.h>
#include <fcntl.h>
#include <unistd.h>

#include "sudo_compat.h"
#include "sudo_debug.h"
#include "sudo_fatal.h"
#include "sudo_gettext.h"
#include "sudo_iolog.h"
#include "sudo_util.h"
#ifdef __APPLE__
#include "logging.h"
#include <rootless.h>
#include <System/sys/codesign.h>
#include <sys/csr.h>
#endif /* __APPLE__ */

/*
 * Wrapper for openat(2) that sets umask and retries as iolog_uid/iolog_gid
 * if openat(2) returns EACCES.
 */
int
iolog_openat(int dfd, const char *path, int flags)
{
    const mode_t iolog_filemode = iolog_get_file_mode();
    const mode_t iolog_dirmode = iolog_get_dir_mode();
    mode_t omask = S_IRWXG|S_IRWXO;
    int fd;
    debug_decl(iolog_openat, SUDO_DEBUG_UTIL);

    if (ISSET(flags, O_CREAT)) {
	/* umask must not be more restrictive than the file modes. */
	omask = umask(ACCESSPERMS & ~(iolog_filemode|iolog_dirmode));
    }
    fd = openat(dfd, path, flags, iolog_filemode);
    if (fd == -1 && errno == EACCES) {
	/* Enable write bit if it is missing. */
	struct stat sb;
	if (fstatat(dfd, path, &sb, 0) == 0) {
	    mode_t write_bits = iolog_filemode & (S_IWUSR|S_IWGRP|S_IWOTH);
	    if ((sb.st_mode & write_bits) != write_bits) {
		if (fchmodat(dfd, path, iolog_filemode, 0) == 0)
		    fd = openat(dfd, path, flags, iolog_filemode);
	    }
	}
    }
	
    if (fd == -1 && errno == EACCES) {
	/* Try again as the I/O log owner (for NFS). */
	if (iolog_swapids(false)) {
	    fd = openat(dfd, path, flags, iolog_filemode);
	    if (!iolog_swapids(true)) {
		/* iolog_swapids() warns on error. */
		if (fd != -1) {
		    close(fd);
		    fd = -1;
		}
	    }
	}
    }
    if (ISSET(flags, O_CREAT))
	umask(omask);
	
#ifdef __APPLE__
    if (fd && (ISSET(flags, O_WRONLY) || ISSET(flags, O_RDWR))) {
	// verify that real opened path for write is not under SIP
	    int csflags = 0;
	    int rv = 0;
	    pid_t pid = getpid();
	    rv = csops(pid, CS_OPS_STATUS, &csflags, sizeof(csflags));
	    if (rv == 0 && (csflags & CS_INSTALLER) != 0) {
		// we are in the installer context
		// check if SIP is on and filesystem path is protected
		bool sipEnabled = (csr_check(CSR_ALLOW_UNRESTRICTED_FS) != 0);
		bool pathProtected = (rootless_check_trusted_fd(fd) == 0);
		if (sipEnabled && pathProtected) {
		    log_warningx(SLOG_NO_STDERR|SLOG_AUDIT, N_("Protected logpath detected"));
		    sudo_warnx(U_("%s: %s %s"), __func__, U_("Protected logpath detected"), path);
		    close(fd);
		    fd = -1;
		}
	    }
    }
#endif /* __APPLE__ */
    debug_return_int(fd);
}
