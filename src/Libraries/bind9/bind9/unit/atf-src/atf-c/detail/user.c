/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 12, 2024.
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
#include <sys/param.h>
#include <sys/types.h>
#include <limits.h>
#include <unistd.h>

#include "sanity.h"
#include "user.h"

/* ---------------------------------------------------------------------
 * Free functions.
 * --------------------------------------------------------------------- */

uid_t
atf_user_euid(void)
{
    return geteuid();
}

bool
atf_user_is_member_of_group(gid_t gid)
{
    static gid_t groups[NGROUPS_MAX];
    static int ngroups = -1;
    bool found;
    int i;

    if (ngroups == -1) {
        ngroups = getgroups(NGROUPS_MAX, groups);
        INV(ngroups >= 0);
    }

    found = false;
    for (i = 0; !found && i < ngroups; i++)
        if (groups[i] == gid)
            found = true;
    return found;
}

bool
atf_user_is_root(void)
{
    return geteuid() == 0;
}

bool
atf_user_is_unprivileged(void)
{
    return geteuid() != 0;
}
