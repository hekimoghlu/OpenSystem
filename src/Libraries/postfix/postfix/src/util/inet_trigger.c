/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 13, 2023.
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

struct inet_trigger {
    int     fd;
    char   *service;
};

/* inet_trigger_event - disconnect from peer */

static void inet_trigger_event(int event, void *context)
{
    struct inet_trigger *ip = (struct inet_trigger *) context;
    static const char *myname = "inet_trigger_event";

    /*
     * Disconnect.
     */
    if (event == EVENT_TIME)
	msg_warn("%s: read timeout for service %s", myname, ip->service);
    event_disable_readwrite(ip->fd);
    event_cancel_timer(inet_trigger_event, context);
    if (close(ip->fd) < 0)
	msg_warn("%s: close %s: %m", myname, ip->service);
    myfree(ip->service);
    myfree((void *) ip);
}


/* inet_trigger - wakeup INET-domain server */

int     inet_trigger(const char *service, const char *buf, ssize_t len, int timeout)
{
    const char *myname = "inet_trigger";
    struct inet_trigger *ip;
    int     fd;

    if (msg_verbose > 1)
	msg_info("%s: service %s", myname, service);

    /*
     * Connect...
     */
    if ((fd = inet_connect(service, BLOCKING, timeout)) < 0) {
	if (msg_verbose)
	    msg_warn("%s: connect to %s: %m", myname, service);
	return (-1);
    }
    close_on_exec(fd, CLOSE_ON_EXEC);
    ip = (struct inet_trigger *) mymalloc(sizeof(*ip));
    ip->fd = fd;
    ip->service = mystrdup(service);

    /*
     * Write the request...
     */
    if (write_buf(fd, buf, len, timeout) < 0
	|| write_buf(fd, "", 1, timeout) < 0)
	if (msg_verbose)
	    msg_warn("%s: write to %s: %m", myname, service);

    /*
     * Wakeup when the peer disconnects, or when we lose patience.
     */
    if (timeout > 0)
	event_request_timer(inet_trigger_event, (void *) ip, timeout + 100);
    event_enable_read(fd, inet_trigger_event, (void *) ip);
    return (0);
}
