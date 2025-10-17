/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 20, 2024.
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

#include <sys_defs.h>
#include <unistd.h>

/* Utility library. */

#include "msg.h"
#include "set_eugid.h"
#include "dot_lockfile.h"
#include "dot_lockfile_as.h"

/* dot_lockfile_as - dotlock file as user */

int     dot_lockfile_as(const char *path, VSTRING *why, uid_t euid, gid_t egid)
{
    uid_t   saved_euid = geteuid();
    gid_t   saved_egid = getegid();
    int     result;

    /*
     * Switch to the target user privileges.
     */
    set_eugid(euid, egid);

    /*
     * Lock that file.
     */
    result = dot_lockfile(path, why);

    /*
     * Restore saved privileges.
     */
    set_eugid(saved_euid, saved_egid);

    return (result);
}

/* dot_unlockfile_as - dotlock file as user */

void     dot_unlockfile_as(const char *path, uid_t euid, gid_t egid)
{
    uid_t   saved_euid = geteuid();
    gid_t   saved_egid = getegid();

    /*
     * Switch to the target user privileges.
     */
    set_eugid(euid, egid);

    /*
     * Lock that file.
     */
    dot_unlockfile(path);

    /*
     * Restore saved privileges.
     */
    set_eugid(saved_euid, saved_egid);
}
