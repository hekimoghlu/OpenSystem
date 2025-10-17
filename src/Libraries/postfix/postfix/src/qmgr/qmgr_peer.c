/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 27, 2023.
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
#include <htable.h>
#include <mymalloc.h>

/* Application-specific. */

#include "qmgr.h"

/* qmgr_peer_create - create and initialize message peer structure */

QMGR_PEER *qmgr_peer_create(QMGR_JOB *job, QMGR_QUEUE *queue)
{
    QMGR_PEER *peer;

    peer = (QMGR_PEER *) mymalloc(sizeof(QMGR_PEER));
    peer->queue = queue;
    peer->job = job;
    QMGR_LIST_APPEND(job->peer_list, peer, peers);
    htable_enter(job->peer_byname, queue->name, (void *) peer);
    peer->refcount = 0;
    QMGR_LIST_INIT(peer->entry_list);
    return (peer);
}

/* qmgr_peer_free - release peer structure */

void    qmgr_peer_free(QMGR_PEER *peer)
{
    const char *myname = "qmgr_peer_free";
    QMGR_JOB *job = peer->job;
    QMGR_QUEUE *queue = peer->queue;

    /*
     * Sanity checks. It is an error to delete a referenced peer structure.
     */
    if (peer->refcount != 0)
	msg_panic("%s: refcount: %d", myname, peer->refcount);
    if (peer->entry_list.next != 0)
	msg_panic("%s: entry list not empty: %s", myname, queue->name);

    QMGR_LIST_UNLINK(job->peer_list, QMGR_PEER *, peer, peers);
    htable_delete(job->peer_byname, queue->name, (void (*) (void *)) 0);
    myfree((void *) peer);
}

/* qmgr_peer_find - lookup peer associated with given job and queue */

QMGR_PEER *qmgr_peer_find(QMGR_JOB *job, QMGR_QUEUE *queue)
{
    return ((QMGR_PEER *) htable_find(job->peer_byname, queue->name));
}

/* qmgr_peer_obtain - find/create peer associated with given job and queue */

QMGR_PEER *qmgr_peer_obtain(QMGR_JOB *job, QMGR_QUEUE *queue)
{
    QMGR_PEER *peer;

    if ((peer = qmgr_peer_find(job, queue)) == 0)
	peer = qmgr_peer_create(job, queue);
    return (peer);
}

/* qmgr_peer_select - select next peer suitable for delivery within given job */

QMGR_PEER *qmgr_peer_select(QMGR_JOB *job)
{
    QMGR_PEER *peer;
    QMGR_QUEUE *queue;

    /*
     * If we find a suitable site, rotate the list to enforce round-robin
     * selection. See similar selection code in qmgr_transport_select().
     */
    for (peer = job->peer_list.next; peer; peer = peer->peers.next) {
	queue = peer->queue;
	if (queue->window > queue->busy_refcount && peer->entry_list.next != 0) {
	    QMGR_LIST_ROTATE(job->peer_list, peer, peers);
	    if (msg_verbose)
		msg_info("qmgr_peer_select: %s %s %s (%d of %d)",
		job->message->queue_id, queue->transport->name, queue->name,
			 queue->busy_refcount + 1, queue->window);
	    return (peer);
	}
    }
    return (0);
}
