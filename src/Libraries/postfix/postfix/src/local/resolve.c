/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 12, 2022.
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
#include <vstring.h>
#include <htable.h>

/* Global library. */

#include <mail_proto.h>
#include <resolve_clnt.h>
#include <rewrite_clnt.h>
#include <tok822.h>
#include <mail_params.h>
#include <defer.h>

/* Application-specific. */

#include "local.h"

/* deliver_resolve_addr - resolve and deliver */

int     deliver_resolve_addr(LOCAL_STATE state, USER_ATTR usr_attr, char *addr)
{
    TOK822 *tree;
    int     result;

    tree = tok822_scan_addr(addr);
    result = deliver_resolve_tree(state, usr_attr, tree);
    tok822_free_tree(tree);
    return (result);
}

/* deliver_resolve_tree - resolve and deliver */

int     deliver_resolve_tree(LOCAL_STATE state, USER_ATTR usr_attr, TOK822 *addr)
{
    const char *myname = "deliver_resolve_tree";
    RESOLVE_REPLY reply;
    int     status;
    ssize_t ext_len;
    char   *ratsign;
    int     rcpt_delim;

    /*
     * Make verbose logging easier to understand.
     */
    state.level++;
    if (msg_verbose)
	MSG_LOG_STATE(myname, state);

    /*
     * Initialize.
     */
    resolve_clnt_init(&reply);

    /*
     * Rewrite the address to canonical form, just like the cleanup service
     * does. Then, resolve the address to (transport, nexhop, recipient),
     * just like the queue manager does. The only part missing here is the
     * virtual address substitution. Message forwarding fixes most of that.
     */
    tok822_rewrite(addr, REWRITE_CANON);
    tok822_resolve(addr, &reply);

    /*
     * First, a healthy portion of error handling.
     */
    if (reply.flags & RESOLVE_FLAG_FAIL) {
	dsb_simple(state.msg_attr.why, "4.3.0", "address resolver failure");
	status = defer_append(BOUNCE_FLAGS(state.request),
			      BOUNCE_ATTR(state.msg_attr));
    } else if (reply.flags & RESOLVE_FLAG_ERROR) {
	dsb_simple(state.msg_attr.why, "5.1.3",
		   "bad recipient address syntax: %s", STR(reply.recipient));
	status = bounce_append(BOUNCE_FLAGS(state.request),
			       BOUNCE_ATTR(state.msg_attr));
    } else {

	/*
	 * Splice in the optional unmatched address extension.
	 */
	if (state.msg_attr.unmatched) {
	    rcpt_delim = state.msg_attr.local[strlen(state.msg_attr.user)];
	    if ((ratsign = strrchr(STR(reply.recipient), '@')) == 0) {
		VSTRING_ADDCH(reply.recipient, rcpt_delim);
		vstring_strcat(reply.recipient, state.msg_attr.unmatched);
	    } else {
		ext_len = strlen(state.msg_attr.unmatched);
		VSTRING_SPACE(reply.recipient, ext_len + 2);
		if ((ratsign = strrchr(STR(reply.recipient), '@')) == 0)
		    msg_panic("%s: recipient @ botch", myname);
		memmove(ratsign + ext_len + 1, ratsign, strlen(ratsign) + 1);
		*ratsign = rcpt_delim;
		memcpy(ratsign + 1, state.msg_attr.unmatched, ext_len);
		VSTRING_SKIP(reply.recipient);
	    }
	}
	state.msg_attr.rcpt.address = STR(reply.recipient);

	/*
	 * Delivery to a local or non-local address. For a while there was
	 * some ugly code to force local recursive alias expansions on a host
	 * with no authority over the local domain, but that code was just
	 * too unclean.
	 */
	if (strcmp(state.msg_attr.relay, STR(reply.transport)) == 0) {
	    status = deliver_recipient(state, usr_attr);
	} else {
	    status = deliver_indirect(state);
	}
    }

    /*
     * Cleanup.
     */
    resolve_clnt_free(&reply);

    return (status);
}
