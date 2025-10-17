/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 20, 2023.
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
#include "set_ugid.h"

/* set_ugid - set real, effective and saved user and group attributes */

void    set_ugid(uid_t uid, gid_t gid)
{
    int     saved_errno = errno;

    if (geteuid() != 0)
	if (seteuid(0) < 0)
	    msg_fatal("seteuid(0): %m");
    if (setgid(gid) < 0)
	msg_fatal("setgid(%ld): %m", (long) gid);
    if (setgroups(1, &gid) < 0)
	msg_fatal("setgroups(1, &%ld): %m", (long) gid);
    if (setuid(uid) < 0)
	msg_fatal("setuid(%ld): %m", (long) uid);
    if (msg_verbose > 1)
	msg_info("setugid: uid %ld gid %ld", (long) uid, (long) gid);
    errno = saved_errno;
}
