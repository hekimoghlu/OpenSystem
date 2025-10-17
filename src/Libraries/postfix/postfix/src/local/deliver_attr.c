/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 7, 2022.
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

/* Utility library. */

#include <msg.h>
#include <vstream.h>
#include <vstring.h>

/* Application-specific. */

#include "local.h"

/* deliver_attr_init - set message delivery attributes to all-zero state */

void    deliver_attr_init(DELIVER_ATTR *attrp)
{
    attrp->level = 0;
    attrp->fp = 0;
    attrp->queue_name = 0;
    attrp->queue_id = 0;
    attrp->offset = 0;
    attrp->sender = 0;
    RECIPIENT_ASSIGN(&(attrp->rcpt), 0, 0, 0, 0, 0);
    attrp->domain = 0;
    attrp->local = 0;
    attrp->user = 0;
    attrp->extension = 0;
    attrp->unmatched = 0;
    attrp->owner = 0;
    attrp->delivered = 0;
    attrp->relay = 0;
    attrp->exp_type = 0;
    attrp->exp_from = 0;
    attrp->why = dsb_create();
}

/* deliver_attr_dump - log message delivery attributes */

void    deliver_attr_dump(DELIVER_ATTR *attrp)
{
    msg_info("level: %d", attrp->level);
    msg_info("path: %s", VSTREAM_PATH(attrp->fp));
    msg_info("fp: 0x%lx", (long) attrp->fp);
    msg_info("queue_name: %s", attrp->queue_name ? attrp->queue_name : "null");
    msg_info("queue_id: %s", attrp->queue_id ? attrp->queue_id : "null");
    msg_info("offset: %ld", attrp->rcpt.offset);
    msg_info("sender: %s", attrp->sender ? attrp->sender : "null");
    msg_info("recipient: %s", attrp->rcpt.address ? attrp->rcpt.address : "null");
    msg_info("domain: %s", attrp->domain ? attrp->domain : "null");
    msg_info("local: %s", attrp->local ? attrp->local : "null");
    msg_info("user: %s", attrp->user ? attrp->user : "null");
    msg_info("extension: %s", attrp->extension ? attrp->extension : "null");
    msg_info("unmatched: %s", attrp->unmatched ? attrp->unmatched : "null");
    msg_info("owner: %s", attrp->owner ? attrp->owner : "null");
    msg_info("delivered: %s", attrp->delivered ? attrp->delivered : "null");
    msg_info("relay: %s", attrp->relay ? attrp->relay : "null");
    msg_info("exp_type: %d", attrp->exp_type);
    msg_info("exp_from: %s", attrp->exp_from ? attrp->exp_from : "null");
    msg_info("why: %s", attrp->why ? "buffer" : "null");
}

/* deliver_attr_free - release storage */

void    deliver_attr_free(DELIVER_ATTR *attrp)
{
    dsb_free(attrp->why);
}
