/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 26, 2021.
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

#include <mymalloc.h>
#include <stringops.h>

/* Global library. */

/* Application-specific. */

#include "qmgr.h"

/* qmgr_error_transport - look up error transport for specified service */

QMGR_TRANSPORT *qmgr_error_transport(const char *service)
{
    QMGR_TRANSPORT *transport;

    /*
     * Find or create retry transport.
     */
    if ((transport = qmgr_transport_find(service)) == 0)
	transport = qmgr_transport_create(service);
    if (QMGR_TRANSPORT_THROTTLED(transport))
	return (0);

    /*
     * Done.
     */
    return (transport);
}

/* qmgr_error_queue - look up error queue for specified service and problem */

QMGR_QUEUE *qmgr_error_queue(const char *service, DSN *dsn)
{
    QMGR_TRANSPORT *transport;
    QMGR_QUEUE *queue;
    char   *nexthop;

    /*
     * Find or create transport.
     */
    if ((transport = qmgr_error_transport(service)) == 0)
	return (0);

    /*
     * Find or create queue.
     */
    nexthop = qmgr_error_nexthop(dsn);
    if ((queue = qmgr_queue_find(transport, nexthop)) == 0)
	queue = qmgr_queue_create(transport, nexthop, nexthop);
    myfree(nexthop);
    if (QMGR_QUEUE_THROTTLED(queue))
	return (0);

    /*
     * Done.
     */
    return (queue);
}

/* qmgr_error_nexthop - compute next-hop information from problem description */

char   *qmgr_error_nexthop(DSN *dsn)
{
    char   *nexthop;

    nexthop = concatenate(dsn->status, " ", dsn->reason, (char *) 0);
    return (nexthop);
}
