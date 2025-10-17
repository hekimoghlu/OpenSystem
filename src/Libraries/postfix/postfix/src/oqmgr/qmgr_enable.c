/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 5, 2022.
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

/* Utility library. */

#include <msg.h>
#include <vstream.h>

/* Application-specific. */

#include "qmgr.h"

/* qmgr_enable_all - enable transports and queues */

void    qmgr_enable_all(void)
{
    QMGR_TRANSPORT *xport;

    if (msg_verbose)
	msg_info("qmgr_enable_all");

    /*
     * The number of transports does not change as a side effect, so this can
     * be a straightforward loop.
     */
    for (xport = qmgr_transport_list.next; xport; xport = xport->peers.next)
	qmgr_enable_transport(xport);
}

/* qmgr_enable_transport - defer todo entries for named transport */

void    qmgr_enable_transport(QMGR_TRANSPORT *transport)
{
    QMGR_QUEUE *queue;
    QMGR_QUEUE *next;

    /*
     * Proceed carefully. Queues may disappear as a side effect.
     */
    if (transport->flags & QMGR_TRANSPORT_STAT_DEAD) {
	if (msg_verbose)
	    msg_info("enable transport %s", transport->name);
	qmgr_transport_unthrottle(transport);
    }
    for (queue = transport->queue_list.next; queue; queue = next) {
	next = queue->peers.next;
	qmgr_enable_queue(queue);
    }
}

/* qmgr_enable_queue - enable and possibly delete queue */

void    qmgr_enable_queue(QMGR_QUEUE *queue)
{
    if (QMGR_QUEUE_THROTTLED(queue)) {
	if (msg_verbose)
	    msg_info("enable site %s/%s", queue->transport->name, queue->name);
	qmgr_queue_unthrottle(queue);
    }
    if (QMGR_QUEUE_READY(queue) && queue->todo.next == 0 && queue->busy.next == 0)
	qmgr_queue_done(queue);
}
