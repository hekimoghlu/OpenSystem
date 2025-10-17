/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 28, 2025.
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
#include <fcntl.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdarg.h>
#include <errno.h>

/* Utility library. */

#include <msg.h>
#include <vstream.h>
#include <connect.h>
#include <mymalloc.h>
#include <iostuff.h>
#include <stringops.h>

/* Global library. */

#include "timed_ipc.h"
#include "mail_proto.h"

/* mail_connect - connect to mail subsystem */

VSTREAM *mail_connect(const char *class, const char *name, int block_mode)
{
    char   *path;
    VSTREAM *stream;
    int     fd;
    char   *sock_name;

    path = mail_pathname(class, name);
    if ((fd = LOCAL_CONNECT(path, block_mode, 0)) < 0) {
	if (msg_verbose)
	    msg_info("connect to subsystem %s: %m", path);
	stream = 0;
    } else {
	if (msg_verbose)
	    msg_info("connect to subsystem %s", path);
	stream = vstream_fdopen(fd, O_RDWR);
	timed_ipc_setup(stream);
	sock_name = concatenate(path, " socket", (char *) 0);
	vstream_control(stream,
			CA_VSTREAM_CTL_PATH(sock_name),
			CA_VSTREAM_CTL_END);
	myfree(sock_name);
    }
    myfree(path);
    return (stream);
}

/* mail_connect_wait - connect to mail service until it succeeds */

VSTREAM *mail_connect_wait(const char *class, const char *name)
{
    VSTREAM *stream;
    int     count = 0;

    /*
     * XXX Solaris workaround for ECONNREFUSED on a busy socket.
     */
    while ((stream = mail_connect(class, name, BLOCKING)) == 0) {
	if (count++ >= 10) {
	    msg_fatal("connect #%d to subsystem %s/%s: %m",
		      count, class, name);
	} else {
	    msg_warn("connect #%d to subsystem %s/%s: %m",
		     count, class, name);
	}
	sleep(10);				/* XXX make configurable */
    }
    return (stream);
}
