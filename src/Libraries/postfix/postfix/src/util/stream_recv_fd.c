/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 26, 2025.
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
#include <errno.h>
#include <stropts.h>
#include <fcntl.h>

#endif

/* Utility library. */

#include <msg.h>
#include <iostuff.h>

/* stream_recv_fd - receive file descriptor */

int     stream_recv_fd(int fd)
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

#ifdef TEST

 /*
  * Proof-of-concept program. Receive a descriptor (presumably from the
  * stream_send_fd test program) and copy its content until EOF.
  */
#include <unistd.h>
#include <string.h>
#include <stdlib.h>
#include <split_at.h>
#include <listen.h>

int     main(int argc, char **argv)
{
    char   *transport;
    char   *endpoint;
    int     listen_sock;
    int     client_sock;
    int     client_fd;
    ssize_t read_count;
    char    buf[1024];

    if (argc != 2
	|| (endpoint = split_at(transport = argv[1], ':')) == 0
	|| *endpoint == 0 || *transport == 0)
	msg_fatal("usage: %s transport:endpoint", argv[0]);

    if (strcmp(transport, "stream") == 0) {
	listen_sock = stream_listen(endpoint, BLOCKING, 0);
    } else {
	msg_fatal("invalid transport name: %s", transport);
    }
    if (listen_sock < 0)
	msg_fatal("listen %s:%s: %m", transport, endpoint);

    client_sock = stream_accept(listen_sock);
    if (client_sock < 0)
	msg_fatal("stream_accept: %m");

    while ((client_fd = stream_recv_fd(client_sock)) >= 0) {
	msg_info("client_fd = %d", client_fd);
	while ((read_count = read(client_fd, buf, sizeof(buf))) > 0)
	    write(1, buf, read_count);
	if (read_count < 0)
	    msg_fatal("read: %m");
	if (close(client_fd) != 0)
	    msg_fatal("close(%d): %m", client_fd);
    }
    exit(0);
}

#endif
