/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 21, 2021.
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

#include <sys/types.h>
#include <sys/stat.h>
#include <stdio.h>
#include <stdlib.h>
#ifdef HAVE_STDBOOL_H
# include <stdbool.h>
#else
# include "compat/stdbool.h"
#endif
#include <time.h>
#include <errno.h>

#include "sudo_compat.h"
#include "sudo_debug.h"
#include "sudo_fatal.h"
#include "sudo_gettext.h"
#include "sudo_util.h"
#include "sudo_iolog.h"

/*
 * Create temporary directory and any parent directories as needed.
 */
bool
iolog_mkdtemp(char *path)
{
    const mode_t iolog_filemode = iolog_get_file_mode();
    const mode_t iolog_dirmode = iolog_get_dir_mode();
    const uid_t iolog_uid = iolog_get_uid();
    const gid_t iolog_gid = iolog_get_gid();
    bool ok = false, uid_changed = false;
    char *dir = sudo_basename(path);
    mode_t omask;
    int dfd;
    debug_decl(iolog_mkdtemp, SUDO_DEBUG_UTIL);

    /* umask must not be more restrictive than the file modes. */
    omask = umask(ACCESSPERMS & ~(iolog_filemode|iolog_dirmode));

    dfd = sudo_open_parent_dir(path, iolog_uid, iolog_gid, iolog_dirmode, true);
    if (dfd == -1 && errno == EACCES) {
	/* Try again as the I/O log owner (for NFS). */
	uid_changed = iolog_swapids(false);
	if (uid_changed)
	    dfd = sudo_open_parent_dir(path, -1, -1, iolog_dirmode, false);
    }
    if (dfd != -1) {
	/* Create final path component. */
	sudo_debug_printf(SUDO_DEBUG_DEBUG|SUDO_DEBUG_LINENO,
	    "mkdtemp %s", path);
	/* We cannot retry mkdtemp() so always open as iolog user */
	if (!uid_changed)
	    uid_changed = iolog_swapids(false);
	if (mkdtempat(dfd, dir) == NULL) {
	    sudo_warn(U_("unable to mkdir %s"), path);
	    ok = false;
	} else {
	    if (fchmodat(dfd, dir, iolog_dirmode, 0) != 0) {
		/* Not a fatal error, pre-existing mode is 0700. */
		sudo_warn(U_("unable to change mode of %s to 0%o"),
		    path, (unsigned int)iolog_dirmode);
	    }
	    ok = true;
	}
	close(dfd);
    }

    umask(omask);

    if (uid_changed) {
	if (!iolog_swapids(true))
	    ok = false;
    }
    debug_return_bool(ok);
}
