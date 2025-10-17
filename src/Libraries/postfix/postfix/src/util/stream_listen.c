/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 26, 2024.
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
/* System interfaces. */

#include <sys_defs.h>

#ifdef STREAM_CONNECTIONS

#include <sys/stat.h>
#include <unistd.h>
#include <errno.h>
#include <stropts.h>
#include <fcntl.h>

#endif

/* Utility library. */

#include "msg.h"
#include "listen.h"

/* stream_listen - create stream listener */

int     stream_listen(const char *path, int unused_backlog, int block_mode)
{
#ifdef STREAM_CONNECTIONS

    /*
     * We can't specify a listen backlog, however, sending file descriptors
     * across a FIFO gives us a backlog buffer of 460 on Solaris 2.4/SPARC.
     */
    return (fifo_listen(path, 0622, block_mode));
#else
    msg_fatal("stream connections are not implemented");
#endif
}

/* stream_accept - accept stream connection */

int     stream_accept(int fd)
{
#ifdef STREAM_CONNECTIONS
    struct strrecvfd fdinfo;

    /*
     * This will return EAGAIN on a non-blocking stream when someone else
     * snatched the connection from us.
     */
    if (ioctl(fd, I_RECVFD, &fdinfo) < 0)
	return (-1);
    return (fdinfo.fd);
#else
            msg_fatal("stream connections are not implemented");
#endif
}
