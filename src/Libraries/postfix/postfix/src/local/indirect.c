/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 13, 2022.
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
#include <htable.h>

/* Global library. */

#include <mail_params.h>
#include <bounce.h>
#include <defer.h>
#include <been_here.h>
#include <sent.h>

/* Application-specific. */

#include "local.h"

/* deliver_indirect - deliver mail via forwarding service */

int     deliver_indirect(LOCAL_STATE state)
{

    /*
     * Suppress duplicate expansion results. Add some sugar to the name to
     * avoid collisions with other duplicate filters. Allow the user to
     * specify an upper bound on the size of the duplicate filter, so that we
     * can handle huge mailing lists with millions of recipients.
     */
    if (msg_verbose)
	msg_info("deliver_indirect: %s", state.msg_attr.rcpt.address);
    if (been_here(state.dup_filter, "indirect %s",
		  state.msg_attr.rcpt.address))
	return (0);

    /*
     * Don't forward a trace-only request.
     */
    if (DEL_REQ_TRACE_ONLY(state.request->flags)) {
	dsb_simple(state.msg_attr.why, "2.0.0", "forwards to %s",
		   state.msg_attr.rcpt.address);
	return (sent(BOUNCE_FLAGS(state.request), SENT_ATTR(state.msg_attr)));
    }

    /*
     * Send the address to the forwarding service. Inherit the delivered
     * attribute from the alias or from the .forward file owner.
     */
    if (forward_append(state.msg_attr)) {
	dsb_simple(state.msg_attr.why, "4.3.0", "unable to forward message");
	return (defer_append(BOUNCE_FLAGS(state.request),
			     BOUNCE_ATTR(state.msg_attr)));
    }
    return (0);
}
