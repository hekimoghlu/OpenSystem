/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 7, 2024.
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
#include <sys/socket.h>
#include <unistd.h>
#include <string.h>

/* Utility library. */

#include <msg.h>
#include <connect.h>
#include <iostuff.h>
#include <mymalloc.h>
#include <events.h>
#include <trigger.h>

struct pass_trigger {
    int     connect_fd;
    char   *service;
    int     pass_fd[2];
};

/* pass_trigger_event - disconnect from peer */

static void pass_trigger_event(int event, void *context)
{
    struct pass_trigger *pp = (struct pass_trigger *) context;
    static const char *myname = "pass_trigger_event";

    /*
     * Disconnect.
     */
    if (event == EVENT_TIME)
	msg_warn("%s: read timeout for service %s", myname, pp->service);
    event_disable_readwrite(pp->connect_fd);
    event_cancel_timer(pass_trigger_event, context);
    /* Don't combine multiple close() calls into one boolean expression. */
    if (close(pp->connect_fd) < 0)
	msg_warn("%s: close %s: %m", myname, pp->service);
    if (close(pp->pass_fd[0]) < 0)
	msg_warn("%s: close pipe: %m", myname);
    if (close(pp->pass_fd[1]) < 0)
	msg_warn("%s: close pipe: %m", myname);
    myfree(pp->service);
    myfree((void *) pp);
}

/* pass_trigger - wakeup local server */

int     pass_trigger(const char *service, const char *buf, ssize_t len, int timeout)
{
    const char *myname = "pass_trigger";
    int     pass_fd[2];
    struct pass_trigger *pp;
    int     connect_fd;

    if (msg_verbose > 1)
	msg_info("%s: service %s", myname, service);

    /*
     * Connect...
     */
    if ((connect_fd = LOCAL_CONNECT(service, BLOCKING, timeout)) < 0) {
	if (msg_verbose)
	    msg_warn("%s: connect to %s: %m", myname, service);
	return (-1);
    }
    close_on_exec(connect_fd, CLOSE_ON_EXEC);

    /*
     * Create a pipe, and send one pipe end to the server.
     */
    if (pipe(pass_fd) < 0)
	msg_fatal("%s: pipe: %m", myname);
    close_on_exec(pass_fd[0], CLOSE_ON_EXEC);
    close_on_exec(pass_fd[1], CLOSE_ON_EXEC);
    if (LOCAL_SEND_FD(connect_fd, pass_fd[0]) < 0)
	msg_fatal("%s: send file descriptor: %m", myname);

    /*
     * Stash away context.
     */
    pp = (struct pass_trigger *) mymalloc(sizeof(*pp));
    pp->connect_fd = connect_fd;
    pp->service = mystrdup(service);
    pp->pass_fd[0] = pass_fd[0];
    pp->pass_fd[1] = pass_fd[1];

    /*
     * Write the request...
     */
    if (write_buf(pass_fd[1], buf, len, timeout) < 0
	|| write_buf(pass_fd[1], "", 1, timeout) < 0)
	if (msg_verbose)
	    msg_warn("%s: write to %s: %m", myname, service);

    /*
     * Wakeup when the peer disconnects, or when we lose patience.
     */
    if (timeout > 0)
	event_request_timer(pass_trigger_event, (void *) pp, timeout + 100);
    event_enable_read(connect_fd, pass_trigger_event, (void *) pp);
    return (0);
}
