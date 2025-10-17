/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 21, 2024.
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
#include <grp.h>
#include <errno.h>

/* Utility library. */

#include "msg.h"
#include "set_eugid.h"

/* set_eugid - set effective user and group attributes */

void    set_eugid(uid_t euid, gid_t egid)
{
    int     saved_errno = errno;

    if (geteuid() != 0)
	if (seteuid(0))
	    msg_fatal("set_eugid: seteuid(0): %m");
    if (setegid(egid) < 0)
	msg_fatal("set_eugid: setegid(%ld): %m", (long) egid);
    if (setgroups(1, &egid) < 0)
	msg_fatal("set_eugid: setgroups(%ld): %m", (long) egid);
    if (euid != 0 && seteuid(euid) < 0)
	msg_fatal("set_eugid: seteuid(%ld): %m", (long) euid);
    if (msg_verbose)
	msg_info("set_eugid: euid %ld egid %ld", (long) euid, (long) egid);
    errno = saved_errno;
}
