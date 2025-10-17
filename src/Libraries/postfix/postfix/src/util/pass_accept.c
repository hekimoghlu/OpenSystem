/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 13, 2024.
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
#include <errno.h>
#include <unistd.h>

/* Utility library. */

#include <msg.h>
#include <listen.h>
#include <attr.h>

#define PASS_ACCEPT_TMOUT	100

/* pass_accept - accept descriptor */

int     pass_accept(int listen_fd)
{
    const char *myname = "pass_accept";
    int     accept_fd;
    int     recv_fd = -1;

    accept_fd = LOCAL_ACCEPT(listen_fd);
    if (accept_fd < 0) {
	if (errno != EAGAIN)
	    msg_warn("%s: cannot accept connection: %m", myname);
	return (-1);
    } else {
	if (read_wait(accept_fd, PASS_ACCEPT_TMOUT) < 0)
	    msg_warn("%s: timeout receiving file descriptor: %m", myname);
	else if ((recv_fd = LOCAL_RECV_FD(accept_fd)) < 0)
	    msg_warn("%s: cannot receive file descriptor: %m", myname);
	if (close(accept_fd) < 0)
	    msg_warn("%s: close: %m", myname);
	return (recv_fd);
    }
}

/* pass_accept_attr - accept descriptor and attribute list */

int     pass_accept_attr(int listen_fd, HTABLE **attr)
{
    const char *myname = "pass_accept_attr";
    int     accept_fd;
    int     recv_fd = -1;

    *attr = 0;
    accept_fd = LOCAL_ACCEPT(listen_fd);
    if (accept_fd < 0) {
	if (errno != EAGAIN)
	    msg_warn("%s: cannot accept connection: %m", myname);
	return (-1);
    } else {
	if (read_wait(accept_fd, PASS_ACCEPT_TMOUT) < 0)
	    msg_warn("%s: timeout receiving file descriptor: %m", myname);
	else if ((recv_fd = LOCAL_RECV_FD(accept_fd)) < 0)
	    msg_warn("%s: cannot receive file descriptor: %m", myname);
	else if (read_wait(accept_fd, PASS_ACCEPT_TMOUT) < 0
	     || recv_pass_attr(accept_fd, attr, PASS_ACCEPT_TMOUT, 0) < 0) {
	    msg_warn("%s: cannot receive connection attributes: %m", myname);
	    if (close(recv_fd) < 0)
		msg_warn("%s: close: %m", myname);
	    recv_fd = -1;
	}
	if (close(accept_fd) < 0)
	    msg_warn("%s: close: %m", myname);
	return (recv_fd);
    }
}
