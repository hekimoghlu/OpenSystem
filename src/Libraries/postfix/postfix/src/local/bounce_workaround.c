/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 12, 2025.
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
#include <strings.h>

/* Utility library. */

#include <msg.h>
#include <mymalloc.h>
#include <vstring.h>
#include <split_at.h>

/* Global library. */

#include <mail_params.h>
#include <strip_addr.h>
#include <stringops.h>
#include <bounce.h>
#include <defer.h>
#include <split_addr.h>
#include <canon_addr.h>

/* Application-specific. */

#include "local.h"

int     bounce_workaround(LOCAL_STATE state)
{
    const char *myname = "bounce_workaround";
    VSTRING *canon_owner = 0;
    int     rcpt_stat;

    /*
     * Look up the substitute sender address.
     */
    if (var_ownreq_special) {
	char   *stripped_recipient;
	char   *owner_alias;
	const char *owner_expansion;

#define FIND_OWNER(lhs, rhs, addr) { \
	lhs = concatenate("owner-", addr, (char *) 0); \
	(void) split_at_right(lhs, '@'); \
	rhs = maps_find(alias_maps, lhs, DICT_FLAG_NONE); \
    }

	FIND_OWNER(owner_alias, owner_expansion, state.msg_attr.rcpt.address);
	if (alias_maps->error == 0 && owner_expansion == 0
	    && (stripped_recipient = strip_addr(state.msg_attr.rcpt.address,
						(char **) 0,
						var_rcpt_delim)) != 0) {
	    myfree(owner_alias);
	    FIND_OWNER(owner_alias, owner_expansion, stripped_recipient);
	    myfree(stripped_recipient);
	}
	if (alias_maps->error == 0 && owner_expansion != 0) {
	    canon_owner = canon_addr_internal(vstring_alloc(10),
					      var_exp_own_alias ?
					      owner_expansion : owner_alias);
	    SET_OWNER_ATTR(state.msg_attr, STR(canon_owner), state.level);
	}
	myfree(owner_alias);
	if (alias_maps->error != 0) {
	    /* At this point, canon_owner == 0. */
	    dsb_simple(state.msg_attr.why, "4.3.0",
		       "alias database unavailable");
	    return (defer_append(BOUNCE_FLAGS(state.request),
				 BOUNCE_ATTR(state.msg_attr)));
	}
    }

    /*
     * Send a delivery status notification with a single recipient to the
     * substitute sender address, before completion of the delivery request.
     */
    if (canon_owner) {
	rcpt_stat =
	    (STR(state.msg_attr.why->status)[0] == '4' ?
	     defer_one : bounce_one)
	    (BOUNCE_FLAGS(state.request),
	     BOUNCE_ONE_ATTR(state.msg_attr));
	vstring_free(canon_owner);
    }

    /*
     * Send a regular delivery status notification, after completion of the
     * delivery request.
     */
    else {
	rcpt_stat =
	    (STR(state.msg_attr.why->status)[0] == '4' ?
	     defer_append : bounce_append)
	    (BOUNCE_FLAGS(state.request),
	     BOUNCE_ATTR(state.msg_attr));
    }
    return (rcpt_stat);
}
