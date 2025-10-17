/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 13, 2022.
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

#include "sys_defs.h"
#include <sys/stat.h>
#include <errno.h>
#include <unistd.h>

/* Utility library. */

#include "msg.h"
#include "sane_fsops.h"
#include "warn_stat.h"

/* sane_link - sanitize link() error returns */

int     sane_link(const char *from, const char *to)
{
    const char *myname = "sane_link";
    int     saved_errno;
    struct stat from_st;
    struct stat to_st;

    /*
     * Normal case: link() succeeds.
     */
    if (link(from, to) >= 0)
	return (0);

    /*
     * Woops. Save errno, and see if the error is an NFS artefact. If it is,
     * pretend the error never happened.
     */
    saved_errno = errno;
    if (stat(from, &from_st) >= 0 && stat(to, &to_st) >= 0
	&& from_st.st_dev == to_st.st_dev
	&& from_st.st_ino == to_st.st_ino) {
	msg_info("%s(%s,%s): worked around spurious NFS error",
		 myname, from, to);
	return (0);
    }

    /*
     * Nope, it didn't. Restore errno and report the error.
     */
    errno = saved_errno;
    return (-1);
}
