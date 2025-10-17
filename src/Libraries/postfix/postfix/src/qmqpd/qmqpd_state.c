/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 23, 2021.
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
#include <time.h>

/* Utility library. */

#include <mymalloc.h>
#include <vstream.h>
#include <vstring.h>

/* Global library. */

#include <mail_stream.h>
#include <cleanup_user.h>
#include <mail_proto.h>

/* Application-specific. */

#include <qmqpd.h>

/* qmqpd_state_alloc - allocate and initialize session state */

QMQPD_STATE *qmqpd_state_alloc(VSTREAM *stream)
{
    QMQPD_STATE *state;

    state = (QMQPD_STATE *) mymalloc(sizeof(*state));
    state->err = CLEANUP_STAT_OK;
    state->client = stream;
    state->message = vstring_alloc(1000);
    state->buf = vstring_alloc(100);
    GETTIMEOFDAY(&state->arrival_time);
    qmqpd_peer_init(state);
    state->queue_id = 0;
    state->cleanup = 0;
    state->dest = 0;
    state->rcpt_count = 0;
    state->reason = 0;
    state->sender = 0;
    state->recipient = 0;
    state->protocol = MAIL_PROTO_QMQP;
    state->where = "initializing client connection";
    state->why_rejected = vstring_alloc(10);
    return (state);
}

/* qmqpd_state_free - destroy session state */

void    qmqpd_state_free(QMQPD_STATE *state)
{
    vstring_free(state->message);
    vstring_free(state->buf);
    qmqpd_peer_reset(state);
    if (state->queue_id)
	myfree(state->queue_id);
    if (state->dest)
	mail_stream_cleanup(state->dest);
    if (state->sender)
	myfree(state->sender);
    if (state->recipient)
	myfree(state->recipient);
    vstring_free(state->why_rejected);
    myfree((void *) state);
}
