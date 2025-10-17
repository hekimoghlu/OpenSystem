/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 18, 2021.
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
#include <mymalloc.h>
#include <stringops.h>

/* Global library. */

#include <bounce.h>

/* Application-specific. */

#include "virtual.h"

/* deliver_recipient - deliver one local recipient */

int     deliver_recipient(LOCAL_STATE state, USER_ATTR usr_attr)
{
    const char *myname = "deliver_recipient";
    VSTRING *folded;
    int     rcpt_stat;

    /*
     * Make verbose logging easier to understand.
     */
    state.level++;
    if (msg_verbose)
	MSG_LOG_STATE(myname, state);

    /*
     * Set up the recipient-specific attributes. The recipient's lookup
     * handle is the full address.
     */
    if (state.msg_attr.delivered == 0)
	state.msg_attr.delivered = state.msg_attr.rcpt.address;
    folded = vstring_alloc(100);
    state.msg_attr.user = casefold(folded, state.msg_attr.rcpt.address);

    /*
     * Deliver
     */
    if (msg_verbose)
	deliver_attr_dump(&state.msg_attr);

    if (deliver_mailbox(state, usr_attr, &rcpt_stat) == 0)
	rcpt_stat = deliver_unknown(state);

    /*
     * Cleanup.
     */
    vstring_free(folded);

    return (rcpt_stat);
}
