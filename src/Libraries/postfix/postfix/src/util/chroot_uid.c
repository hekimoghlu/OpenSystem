/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 19, 2024.
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
#include <pwd.h>
#include <unistd.h>
#include <grp.h>

/* Utility library. */

#include "msg.h"
#include "chroot_uid.h"

/* chroot_uid - restrict the damage that this program can do */

void    chroot_uid(const char *root_dir, const char *user_name)
{
    struct passwd *pwd;
    uid_t   uid;
    gid_t   gid;

    /*
     * Look up the uid/gid before entering the jail, and save them so they
     * can't be clobbered. Set up the primary and secondary groups.
     */
    if (user_name != 0) {
	if ((pwd = getpwnam(user_name)) == 0)
	    msg_fatal("unknown user: %s", user_name);
	uid = pwd->pw_uid;
	gid = pwd->pw_gid;
	if (setgid(gid) < 0)
	    msg_fatal("setgid(%ld): %m", (long) gid);
	if (initgroups(user_name, gid) < 0)
	    msg_fatal("initgroups: %m");
    }

    /*
     * Enter the jail.
     */
    if (root_dir) {
	if (chroot(root_dir))
	    msg_fatal("chroot(%s): %m", root_dir);
	if (chdir("/"))
	    msg_fatal("chdir(/): %m");
    }

    /*
     * Drop the user privileges.
     */
    if (user_name != 0)
	if (setuid(uid) < 0)
	    msg_fatal("setuid(%ld): %m", (long) uid);

    /*
     * Give the desperate developer a clue of what is happening.
     */
    if (msg_verbose > 1)
	msg_info("chroot %s user %s",
		 root_dir ? root_dir : "(none)",
		 user_name ? user_name : "(none)");
}
