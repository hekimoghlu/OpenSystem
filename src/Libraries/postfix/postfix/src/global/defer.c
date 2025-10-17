/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 4, 2022.
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
#include <mail_queue.h>
#include <mail_proto.h>
#include <flush_clnt.h>
#include <verify.h>
#include <dsn_util.h>
#include <rcpt_print.h>
#include <dsn_print.h>
#include <log_adhoc.h>
#include <trace.h>
#include <defer.h>

#define STR(x)	vstring_str(x)

/* defer_append - defer message delivery */

int     defer_append(int flags, const char *id, MSG_STATS *stats,
		             RECIPIENT *rcpt, const char *relay,
		             DSN *dsn)
{
    DSN     my_dsn = *dsn;
    DSN    *dsn_res;

    /*
     * Sanity check.
     */
    if (my_dsn.status[0] != '4' || !dsn_valid(my_dsn.status)) {
	msg_warn("defer_append: ignoring dsn code \"%s\"", my_dsn.status);
	my_dsn.status = "4.0.0";
    }

    /*
     * DSN filter (Postfix 3.0).
     */
    if (delivery_status_filter != 0
    && (dsn_res = dsn_filter_lookup(delivery_status_filter, &my_dsn)) != 0) {
	if (dsn_res->status[0] == '5')
	    return (bounce_append_intern(flags, id, stats, rcpt, relay, dsn_res));
	my_dsn = *dsn_res;
    }
    return (defer_append_intern(flags, id, stats, rcpt, relay, &my_dsn));
}

/* defer_append_intern - defer message delivery */

int     defer_append_intern(int flags, const char *id, MSG_STATS *stats,
			            RECIPIENT *rcpt, const char *relay,
			            DSN *dsn)
{
    const char *rcpt_domain;
    DSN     my_dsn = *dsn;
    int     status;

    /*
     * MTA-requested address verification information is stored in the verify
     * service database.
     */
    if (flags & DEL_REQ_FLAG_MTA_VRFY) {
	my_dsn.action = "undeliverable";
	status = verify_append(id, stats, rcpt, relay, &my_dsn,
			       DEL_RCPT_STAT_DEFER);
	return (status);
    }

    /*
     * User-requested address verification information is logged and mailed
     * to the requesting user.
     */
    if (flags & DEL_REQ_FLAG_USR_VRFY) {
	my_dsn.action = "undeliverable";
	status = trace_append(flags, id, stats, rcpt, relay, &my_dsn);
	return (status);
    }

    /*
     * Normal mail delivery. May also send a delivery record to the user.
     * 
     * XXX DSN We write all deferred recipients to the defer logfile regardless
     * of DSN NOTIFY options, because those options don't apply to mailq(1)
     * reports or to postmaster notifications.
     */
    else {

	/*
	 * Supply default action.
	 */
	my_dsn.action = "delayed";

	if (mail_command_client(MAIL_CLASS_PRIVATE, var_defer_service,
			   SEND_ATTR_INT(MAIL_ATTR_NREQ, BOUNCE_CMD_APPEND),
				SEND_ATTR_INT(MAIL_ATTR_FLAGS, flags),
				SEND_ATTR_STR(MAIL_ATTR_QUEUEID, id),
				SEND_ATTR_FUNC(rcpt_print, (void *) rcpt),
				SEND_ATTR_FUNC(dsn_print, (void *) &my_dsn),
				ATTR_TYPE_END) != 0)
	    msg_warn("%s: %s service failure", id, var_defer_service);
	log_adhoc(id, stats, rcpt, relay, &my_dsn, "deferred");

	/*
	 * Traced delivery.
	 */
	if (flags & DEL_REQ_FLAG_RECORD)
	    if (trace_append(flags, id, stats, rcpt, relay, &my_dsn) != 0)
		msg_warn("%s: %s service failure", id, var_trace_service);

	/*
	 * Notify the fast flush service. XXX Should not this belong in the
	 * bounce/defer daemon? Well, doing it here is more robust.
	 */
	if ((rcpt_domain = strrchr(rcpt->address, '@')) != 0
	    && *++rcpt_domain != 0)
	    switch (flush_add(rcpt_domain, id)) {
	    case FLUSH_STAT_OK:
	    case FLUSH_STAT_DENY:
		break;
	    default:
		msg_warn("%s: %s service failure", id, var_flush_service);
		break;
	    }
	return (-1);
    }
}

/* defer_flush - flush the defer log and deliver to the sender */

int     defer_flush(int flags, const char *queue, const char *id,
		            const char *encoding, int smtputf8,
		            const char *sender, const char *dsn_envid,
		            int dsn_ret)
{
    flags |= BOUNCE_FLAG_DELRCPT;

    if (mail_command_client(MAIL_CLASS_PRIVATE, var_defer_service,
			    SEND_ATTR_INT(MAIL_ATTR_NREQ, BOUNCE_CMD_FLUSH),
			    SEND_ATTR_INT(MAIL_ATTR_FLAGS, flags),
			    SEND_ATTR_STR(MAIL_ATTR_QUEUE, queue),
			    SEND_ATTR_STR(MAIL_ATTR_QUEUEID, id),
			    SEND_ATTR_STR(MAIL_ATTR_ENCODING, encoding),
			    SEND_ATTR_INT(MAIL_ATTR_SMTPUTF8, smtputf8),
			    SEND_ATTR_STR(MAIL_ATTR_SENDER, sender),
			    SEND_ATTR_STR(MAIL_ATTR_DSN_ENVID, dsn_envid),
			    SEND_ATTR_INT(MAIL_ATTR_DSN_RET, dsn_ret),
			    ATTR_TYPE_END) == 0) {
	return (0);
    } else {
	return (-1);
    }
}

/* defer_warn - send a copy of the defer log to the sender as a warning bounce
 * do not flush the log */

int     defer_warn(int flags, const char *queue, const char *id,
		           const char *encoding, int smtputf8,
		         const char *sender, const char *envid, int dsn_ret)
{
    if (mail_command_client(MAIL_CLASS_PRIVATE, var_defer_service,
			    SEND_ATTR_INT(MAIL_ATTR_NREQ, BOUNCE_CMD_WARN),
			    SEND_ATTR_INT(MAIL_ATTR_FLAGS, flags),
			    SEND_ATTR_STR(MAIL_ATTR_QUEUE, queue),
			    SEND_ATTR_STR(MAIL_ATTR_QUEUEID, id),
			    SEND_ATTR_STR(MAIL_ATTR_ENCODING, encoding),
			    SEND_ATTR_INT(MAIL_ATTR_SMTPUTF8, smtputf8),
			    SEND_ATTR_STR(MAIL_ATTR_SENDER, sender),
			    SEND_ATTR_STR(MAIL_ATTR_DSN_ENVID, envid),
			    SEND_ATTR_INT(MAIL_ATTR_DSN_RET, dsn_ret),
			    ATTR_TYPE_END) == 0) {
	return (0);
    } else {
	return (-1);
    }
}

/* defer_one - defer mail for one recipient */

int     defer_one(int flags, const char *queue, const char *id,
		          const char *encoding, int smtputf8,
		          const char *sender, const char *dsn_envid,
		          int dsn_ret, MSG_STATS *stats, RECIPIENT *rcpt,
		          const char *relay, DSN *dsn)
{
    DSN     my_dsn = *dsn;
    DSN    *dsn_res;

    /*
     * Sanity check.
     */
    if (my_dsn.status[0] != '4' || !dsn_valid(my_dsn.status)) {
	msg_warn("defer_one: ignoring dsn code \"%s\"", my_dsn.status);
	my_dsn.status = "4.0.0";
    }

    /*
     * DSN filter (Postfix 3.0).
     */
    if (delivery_status_filter != 0
    && (dsn_res = dsn_filter_lookup(delivery_status_filter, &my_dsn)) != 0) {
	if (dsn_res->status[0] == '5')
	    return (bounce_one_intern(flags, queue, id, encoding, smtputf8,
				      sender, dsn_envid, dsn_ret, stats,
				      rcpt, relay, dsn_res));
	my_dsn = *dsn_res;
    }
    return (defer_append_intern(flags, id, stats, rcpt, relay, &my_dsn));
}
