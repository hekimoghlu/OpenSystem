/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 21, 2024.
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

/* Utility library. */

#include <msg.h>
#include <vstring.h>

/* Global library. */

#define DSN_INTERN
#include <mail_params.h>
#include <verify.h>
#include <log_adhoc.h>
#include <trace.h>
#include <defer.h>
#include <sent.h>
#include <dsn_util.h>
#include <dsn_mask.h>

/* Application-specific. */

/* sent - log that a message was or could be sent */

int     sent(int flags, const char *id, MSG_STATS *stats,
	             RECIPIENT *recipient, const char *relay,
	             DSN *dsn)
{
    DSN     my_dsn = *dsn;
    DSN    *dsn_res;
    int     status;

    /*
     * Sanity check.
     */
    if (my_dsn.status[0] != '2' || !dsn_valid(my_dsn.status)) {
	msg_warn("sent: ignoring dsn code \"%s\"", my_dsn.status);
	my_dsn.status = "2.0.0";
    }

    /*
     * DSN filter (Postfix 3.0).
     */
    if (delivery_status_filter != 0
     && (dsn_res = dsn_filter_lookup(delivery_status_filter, &my_dsn)) != 0)
	my_dsn = *dsn_res;

    /*
     * MTA-requested address verification information is stored in the verify
     * service database.
     */
    if (flags & DEL_REQ_FLAG_MTA_VRFY) {
	my_dsn.action = "deliverable";
	status = verify_append(id, stats, recipient, relay, &my_dsn,
			       DEL_RCPT_STAT_OK);
	return (status);
    }

    /*
     * User-requested address verification information is logged and mailed
     * to the requesting user.
     */
    if (flags & DEL_REQ_FLAG_USR_VRFY) {
	my_dsn.action = "deliverable";
	status = trace_append(flags, id, stats, recipient, relay, &my_dsn);
	return (status);
    }

    /*
     * Normal mail delivery. May also send a delivery record to the user.
     */
    else {

	/* Readability macros: record all deliveries, or the delayed ones. */
#define REC_ALL_SENT(flags) (flags & DEL_REQ_FLAG_RECORD)
#define REC_DLY_SENT(flags, rcpt) \
	((flags & DEL_REQ_FLAG_REC_DLY_SENT) \
	&& (rcpt->dsn_notify == 0 || (rcpt->dsn_notify & DSN_NOTIFY_DELAY)))

	if (my_dsn.action == 0 || my_dsn.action[0] == 0)
	    my_dsn.action = "delivered";

	if (((REC_ALL_SENT(flags) == 0 && REC_DLY_SENT(flags, recipient) == 0)
	  || trace_append(flags, id, stats, recipient, relay, &my_dsn) == 0)
	    && ((recipient->dsn_notify & DSN_NOTIFY_SUCCESS) == 0
	|| trace_append(flags, id, stats, recipient, relay, &my_dsn) == 0)) {
	    log_adhoc(id, stats, recipient, relay, &my_dsn, "sent");
	    status = 0;
	} else {
	    VSTRING *junk = vstring_alloc(100);

	    vstring_sprintf(junk, "%s: %s service failed",
			    id, var_trace_service);
	    my_dsn.reason = vstring_str(junk);
	    my_dsn.status = "4.3.0";
	    status = defer_append(flags, id, stats, recipient, relay, &my_dsn);
	    vstring_free(junk);
	}
	return (status);
    }
}
