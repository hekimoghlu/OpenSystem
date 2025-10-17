/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 5, 2022.
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
#include <string.h>

#ifdef STRCASECMP_IN_STRINGS_H
#include <strings.h>
#endif

/* Utility library. */

#include <msg.h>
#include <stringops.h>
#include <mymalloc.h>
#include <vstring.h>

/* Global library. */

#include <been_here.h>
#include <mail_params.h>
#include <mail_proto.h>
#include <bounce.h>
#include <mail_addr.h>
#include <sent.h>
#include <deliver_pass.h>
#include <defer.h>

/* Application-specific. */

#include "local.h"

/* deliver_unknown - delivery for unknown recipients */

int     deliver_unknown(LOCAL_STATE state, USER_ATTR usr_attr)
{
    const char *myname = "deliver_unknown";
    int     status;
    VSTRING *expand_luser;
    static MAPS *transp_maps;
    const char *map_transport;

    /*
     * Make verbose logging easier to understand.
     */
    state.level++;
    if (msg_verbose)
	MSG_LOG_STATE(myname, state);

    /*
     * DUPLICATE/LOOP ELIMINATION
     * 
     * Don't deliver the same user twice.
     */
    if (been_here(state.dup_filter, "%s %s", myname, state.msg_attr.local))
	return (0);

    /*
     * The fall-back transport specifies a delivery machanism that handles
     * users not found in the aliases or UNIX passwd databases.
     */
    if (*var_fbck_transp_maps && transp_maps == 0)
	transp_maps = maps_create(VAR_FBCK_TRANSP_MAPS, var_fbck_transp_maps,
				  DICT_FLAG_LOCK | DICT_FLAG_NO_REGSUB
				  | DICT_FLAG_UTF8_REQUEST);
    /* The -1 is a hint for the down-stream deliver_completed() function. */
    if (transp_maps
	&& (map_transport = maps_find(transp_maps, state.msg_attr.user,
				      DICT_FLAG_NONE)) != 0) {
	state.msg_attr.rcpt.offset = -1L;
	return (deliver_pass(MAIL_CLASS_PRIVATE, map_transport,
			     state.request, &state.msg_attr.rcpt));
    } else if (transp_maps && transp_maps->error != 0) {
	/* Details in the logfile. */
	dsb_simple(state.msg_attr.why, "4.3.0", "table lookup failure");
	return (defer_append(BOUNCE_FLAGS(state.request),
			     BOUNCE_ATTR(state.msg_attr)));
    }
    if (*var_fallback_transport) {
	state.msg_attr.rcpt.offset = -1L;
	return (deliver_pass(MAIL_CLASS_PRIVATE, var_fallback_transport,
			     state.request, &state.msg_attr.rcpt));
    }

    /*
     * Subject the luser_relay address to $name expansion, disable
     * propagation of unmatched address extension, and re-inject the address
     * into the delivery machinery. Do not give special treatment to "|stuff"
     * or /stuff.
     */
    if (*var_luser_relay) {
	state.msg_attr.unmatched = 0;
	expand_luser = vstring_alloc(100);
	local_expand(expand_luser, var_luser_relay, &state, &usr_attr, (void *) 0);
	status = deliver_resolve_addr(state, usr_attr, STR(expand_luser));
	vstring_free(expand_luser);
	return (status);
    }

    /*
     * If no alias was found for a required reserved name, toss the message
     * into the bit bucket, and issue a warning instead.
     */
#define STREQ(x,y) (strcasecmp(x,y) == 0)

    if (STREQ(state.msg_attr.user, MAIL_ADDR_MAIL_DAEMON)
	|| STREQ(state.msg_attr.user, MAIL_ADDR_POSTMASTER)) {
	msg_warn("required alias not found: %s", state.msg_attr.user);
	dsb_simple(state.msg_attr.why, "2.0.0", "discarded");
	return (sent(BOUNCE_FLAGS(state.request), SENT_ATTR(state.msg_attr)));
    }

    /*
     * Bounce the message when no luser relay is specified.
     */
    dsb_simple(state.msg_attr.why, "5.1.1",
	       "unknown user: \"%s\"", state.msg_attr.user);
    return (bounce_append(BOUNCE_FLAGS(state.request),
			  BOUNCE_ATTR(state.msg_attr)));
}
