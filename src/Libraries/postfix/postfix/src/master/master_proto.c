/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 11, 2022.
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

#include <msg.h>

/* Global library. */

#include "master_proto.h"

int     master_notify(int pid, unsigned generation, int status)
{
    const char *myname = "master_notify";
    MASTER_STATUS stat;

    /*
     * We use a simple binary protocol to minimize security risks. Since this
     * is local IPC, there are no byte order or word length issues. The
     * server treats this information as gossip, so sending a bad PID or a
     * bad status code will only have amusement value.
     */
    stat.pid = pid;
    stat.gen = generation;
    stat.avail = status;

    if (write(MASTER_STATUS_FD, (void *) &stat, sizeof(stat)) != sizeof(stat)) {
	if (msg_verbose)
	    msg_info("%s: status %d: %m", myname, status);
	return (-1);
    } else {
	if (msg_verbose)
	    msg_info("%s: status %d", myname, status);
	return (0);
    }
}
