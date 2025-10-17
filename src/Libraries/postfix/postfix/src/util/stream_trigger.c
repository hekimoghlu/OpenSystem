/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 3, 2024.
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
#include <string.h>

/* Utility library. */

#include <msg.h>
#include <connect.h>
#include <iostuff.h>
#include <mymalloc.h>
#include <events.h>
#include <trigger.h>

struct stream_trigger {
    int     fd;
    char   *service;
};

/* stream_trigger_event - disconnect from peer */

static void stream_trigger_event(int event, void *context)
{
    struct stream_trigger *sp = (struct stream_trigger *) context;
    static const char *myname = "stream_trigger_event";

    /*
     * Disconnect.
     */
    if (event == EVENT_TIME)
	msg_warn("%s: read timeout for service %s", myname, sp->service);
    event_disable_readwrite(sp->fd);
    event_cancel_timer(stream_trigger_event, context);
    if (close(sp->fd) < 0)
	msg_warn("%s: close %s: %m", myname, sp->service);
    myfree(sp->service);
    myfree((void *) sp);
}

/* stream_trigger - wakeup stream server */

int     stream_trigger(const char *service, const char *buf, ssize_t len, int timeout)
{
    const char *myname = "stream_trigger";
    struct stream_trigger *sp;
    int     fd;

    if (msg_verbose > 1)
	msg_info("%s: service %s", myname, service);

    /*
     * Connect...
     */
    if ((fd = stream_connect(service, BLOCKING, timeout)) < 0) {
	if (msg_verbose)
	    msg_warn("%s: connect to %s: %m", myname, service);
	return (-1);
    }
    close_on_exec(fd, CLOSE_ON_EXEC);

    /*
     * Stash away context.
     */
    sp = (struct stream_trigger *) mymalloc(sizeof(*sp));
    sp->fd = fd;
    sp->service = mystrdup(service);

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
	event_request_timer(stream_trigger_event, (void *) sp, timeout + 100);
    event_enable_read(fd, stream_trigger_event, (void *) sp);
    return (0);
}
