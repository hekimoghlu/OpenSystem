/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 1, 2025.
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

#ifdef STREAM_CONNECTIONS

#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <stropts.h>

#endif

/* Utility library. */

#include <msg.h>
#include <connect.h>

/* stream_connect - connect to stream listener */

int     stream_connect(const char *path, int block_mode, int unused_timeout)
{
#ifdef STREAM_CONNECTIONS
    const char *myname = "stream_connect";
    int     pair[2];
    int     fifo;

    /*
     * The requested file system object must exist, otherwise we can't reach
     * the server.
     */
    if ((fifo = open(path, O_WRONLY | O_NONBLOCK, 0)) < 0)
	return (-1);

    /*
     * This is for {unix,inet}_connect() compatibility.
     */
    if (block_mode == BLOCKING)
	non_blocking(fifo, BLOCKING);

    /*
     * Create a pipe, and send one pipe end to the server.
     */
    if (pipe(pair) < 0)
	msg_fatal("%s: pipe: %m", myname);
    if (ioctl(fifo, I_SENDFD, pair[1]) < 0)
	msg_fatal("%s: send file descriptor: %m", myname);
    close(pair[1]);

    /*
     * This is for {unix,inet}_connect() compatibility.
     */
    if (block_mode == NON_BLOCKING)
	non_blocking(pair[0], NON_BLOCKING);

    /*
     * Cleanup.
     */
    close(fifo);

    /*
     * Keep the other end of the pipe.
     */
    return (pair[0]);
#else
    msg_fatal("stream connections are not implemented");
#endif
}
