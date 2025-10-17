/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 22, 2022.
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
#include <unistd.h>
#include <errno.h>
#include <grp.h>
#include <limits.h>

#include "sudo_compat.h"
#include "sudo_debug.h"
#include "sudo_util.h"

int
sudo_setgroups_v1(int ngids, const GETGROUPS_T *gids)
{
    int maxgids, ret;
    debug_decl(sudo_setgroups, SUDO_DEBUG_UTIL);

    ret = setgroups(ngids, (GETGROUPS_T *)gids);
    if (ret == -1 && errno == EINVAL) {
	/* Too many groups, try again with fewer. */
	maxgids = (int)sysconf(_SC_NGROUPS_MAX);
	if (maxgids == -1)
	    maxgids = NGROUPS_MAX;
	if (ngids > maxgids)
	    ret = setgroups(maxgids, (GETGROUPS_T *)gids);
    }
    debug_return_int(ret);
}
