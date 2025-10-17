/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 10, 2025.
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
#include <time.h>
#include <unistd.h>

#include "sudo_compat.h"
#include "sudo_debug.h"
#include "sudo_fatal.h"
#include "sudo_gettext.h"
#include "sudo_iolog.h"

/*
 * Set effective user and group-IDs to iolog_uid and iolog_gid.
 * If restore flag is set, swap them back.
 */
bool
iolog_swapids(bool restore)
{
#ifdef HAVE_SETEUID
    static uid_t user_euid = (uid_t)-1;
    static gid_t user_egid = (gid_t)-1;
    const uid_t iolog_uid = iolog_get_uid();
    const gid_t iolog_gid = iolog_get_gid();
    debug_decl(io_swapids, SUDO_DEBUG_UTIL);

    if (user_euid == (uid_t)-1)
	user_euid = geteuid();
    if (user_egid == (gid_t)-1)
	user_egid = getegid();

    if (restore) {
	if (seteuid(user_euid) == -1) {
	    sudo_debug_printf(SUDO_DEBUG_ERROR|SUDO_DEBUG_ERRNO,
		"%s: unable to restore effective uid to %d", __func__,
		(int)user_euid);
	    sudo_warn("seteuid() %d -> %d", (int)iolog_uid, (int)user_euid);
	    debug_return_bool(false);
	}
	if (setegid(user_egid) == -1) {
	    sudo_debug_printf(SUDO_DEBUG_ERROR|SUDO_DEBUG_ERRNO,
		"%s: unable to restore effective gid to %d", __func__,
		(int)user_egid);
	    sudo_warn("setegid() %d -> %d", (int)iolog_gid, (int)user_egid);
	    debug_return_bool(false);
	}
    } else {
	/* Fail silently if the user has insufficient privileges. */
	if (setegid(iolog_gid) == -1) {
	    sudo_debug_printf(SUDO_DEBUG_ERROR|SUDO_DEBUG_ERRNO,
		"%s: unable to set effective gid to %d", __func__,
		(int)iolog_gid);
	    debug_return_bool(false);
	}
	if (seteuid(iolog_uid) == -1) {
	    sudo_debug_printf(SUDO_DEBUG_ERROR|SUDO_DEBUG_ERRNO,
		"%s: unable to set effective uid to %d", __func__,
		(int)iolog_uid);
	    debug_return_bool(false);
	}
    }
    debug_return_bool(true);
#else
    return false;
#endif
}
