/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 28, 2023.
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

/* Global library. */

#include <mail_proto.h>
#include <defer.h>

/* Application-specific. */

#include "qmgr.h"

/* qmgr_defer_transport - defer todo entries for named transport */

void    qmgr_defer_transport(QMGR_TRANSPORT *transport, DSN *dsn)
{
    QMGR_QUEUE *queue;
    QMGR_QUEUE *next;

    if (msg_verbose)
	msg_info("defer transport %s: %s %s",
		 transport->name, dsn->status, dsn->reason);

    /*
     * Proceed carefully. Queues may disappear as a side effect.
     */
    for (queue = transport->queue_list.next; queue; queue = next) {
	next = queue->peers.next;
	qmgr_defer_todo(queue, dsn);
    }
}

/* qmgr_defer_todo - defer all todo queue entries for specific site */

void    qmgr_defer_todo(QMGR_QUEUE *queue, DSN *dsn)
{
    QMGR_ENTRY *entry;
    QMGR_ENTRY *next;
    QMGR_MESSAGE *message;
    RECIPIENT *recipient;
    int     nrcpt;
    QMGR_QUEUE *retry_queue;

    /*
     * Sanity checks.
     */
    if (msg_verbose)
	msg_info("defer site %s: %s %s",
		 queue->name, dsn->status, dsn->reason);

    /*
     * See if we can redirect the deliveries to the retry(8) delivery agent,
     * so that they can be handled asynchronously. If the retry(8) service is
     * unavailable, use the synchronous defer(8) server. With a large todo
     * queue, this blocks the queue manager for a significant time.
     */
    retry_queue = qmgr_error_queue(MAIL_SERVICE_RETRY, dsn);

    /*
     * Proceed carefully. Queue entries may disappear as a side effect.
     */
    for (entry = queue->todo.next; entry != 0; entry = next) {
	next = entry->peers.next;
	if (retry_queue != 0) {
	    qmgr_entry_move_todo(retry_queue, entry);
	    continue;
	}
	message = entry->message;
	for (nrcpt = 0; nrcpt < entry->rcpt_list.len; nrcpt++) {
	    recipient = entry->rcpt_list.info + nrcpt;
	    qmgr_defer_recipient(message, recipient, dsn);
	}
	qmgr_entry_done(entry, QMGR_QUEUE_TODO);
    }
}

/* qmgr_defer_recipient - defer delivery of specific recipient */

void    qmgr_defer_recipient(QMGR_MESSAGE *message, RECIPIENT *recipient,
			             DSN *dsn)
{
    MSG_STATS stats;

    /*
     * Update the message structure and log the message disposition.
     */
    message->flags |= defer_append(message->tflags, message->queue_id,
				 QMGR_MSG_STATS(&stats, message), recipient,
				   "none", dsn);
}
