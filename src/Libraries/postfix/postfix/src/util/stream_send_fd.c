/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 24, 2022.
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

#include <sys_defs.h>			/* includes <sys/types.h> */

#ifdef STREAM_CONNECTIONS

#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <stropts.h>

#endif

/* Utility library. */

#include <msg.h>
#include <iostuff.h>

/* stream_send_fd - send file descriptor */

int     stream_send_fd(int fd, int sendfd)
{
#ifdef STREAM_CONNECTIONS
    const char *myname = "stream_send_fd";

    if (ioctl(fd, I_SENDFD, sendfd) < 0)
	msg_fatal("%s: send file descriptor %d: %m", myname, sendfd);
    return (0);
#else
            msg_fatal("stream connections are not implemented");
#endif
}

#ifdef TEST

 /*
  * Proof-of-concept program. Open a file and send the descriptor, presumably
  * to the stream_recv_fd test program.
  */
#include <unistd.h>
#include <fcntl.h>
#include <string.h>
#include <stdlib.h>
#include <split_at.h>
#include <connect.h>

int     main(int argc, char **argv)
{
    char   *transport;
    char   *endpoint;
    char   *path;
    int     server_sock;
    int     client_fd;

    if (argc < 3
	|| (endpoint = split_at(transport = argv[1], ':')) == 0
	|| *endpoint == 0 || *transport == 0)
	msg_fatal("usage: %s transport:endpoint file...", argv[0]);

    if (strcmp(transport, "stream") == 0) {
	server_sock = stream_connect(endpoint, BLOCKING, 0);
    } else {
	msg_fatal("invalid transport name: %s", transport);
    }
    if (server_sock < 0)
	msg_fatal("connect %s:%s: %m", transport, endpoint);

    argv += 2;
    while ((path = *argv++) != 0) {
	if ((client_fd = open(path, O_RDONLY, 0)) < 0)
	    msg_fatal("open %s: %m", path);
	msg_info("path=%s client_fd=%d", path, client_fd);
	if (stream_send_fd(server_sock, client_fd) < 0)
	    msg_fatal("send file descriptor: %m");
	if (close(client_fd) != 0)
	    msg_fatal("close(%d): %m", client_fd);
    }
    exit(0);
}

#endif
