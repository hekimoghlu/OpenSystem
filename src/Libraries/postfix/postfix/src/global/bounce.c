/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 6, 2022.
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
#include <mymalloc.h>

/* Global library. */

#define DSN_INTERN
#include <mail_params.h>
#include <mail_proto.h>
#include <log_adhoc.h>
#include <dsn_util.h>
#include <rcpt_print.h>
#include <dsn_print.h>
#include <verify.h>
#include <defer.h>
#include <trace.h>
#include <bounce.h>

/* Shared internally, between bounce and defer clients. */

DSN_FILTER *delivery_status_filter;

/* bounce_append - append delivery status to per-message bounce log */

int     bounce_append(int flags, const char *id, MSG_STATS *stats,
		              RECIPIENT *rcpt, const char *relay,
		              DSN *dsn)
{
    DSN     my_dsn = *dsn;
    DSN    *dsn_res;

    /*
     * Sanity check. If we're really confident, change this into msg_panic
     * (remember, this information may be under control by a hostile server).
     */
    if (my_dsn.status[0] != '5' || !dsn_valid(my_dsn.status)) {
	msg_warn("bounce_append: ignoring dsn code \"%s\"", my_dsn.status);
	my_dsn.status = "5.0.0";
    }

    /*
     * DSN filter (Postfix 3.0).
     */
    if (delivery_status_filter != 0
    && (dsn_res = dsn_filter_lookup(delivery_status_filter, &my_dsn)) != 0) {
	if (dsn_res->status[0] == '4')
	    return (defer_append_intern(flags, id, stats, rcpt, relay, dsn_res));
	my_dsn = *dsn_res;
    }
    return (bounce_append_intern(flags, id, stats, rcpt, relay, &my_dsn));
}

/* bounce_append_intern - append delivery status to per-message bounce log */

int     bounce_append_intern(int flags, const char *id, MSG_STATS *stats,
			             RECIPIENT *rcpt, const char *relay,
			             DSN *dsn)
{
    DSN     my_dsn = *dsn;
    int     status;

    /*
     * MTA-requested address verification information is stored in the verify
     * service database.
     */
    if (flags & DEL_REQ_FLAG_MTA_VRFY) {
	my_dsn.action = "undeliverable";
	status = verify_append(id, stats, rcpt, relay, &my_dsn,
			       DEL_RCPT_STAT_BOUNCE);
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
     * Normal (well almost) delivery. When we're pretending that we can't
     * bounce, don't create a defer log file when we wouldn't keep the bounce
     * log file. That's a lot of negatives in one sentence.
     */
    else if (var_soft_bounce && (flags & BOUNCE_FLAG_CLEAN)) {
	return (-1);
    }

    /*
     * Normal mail delivery. May also send a delivery record to the user.
     * 
     * XXX DSN We write all recipients to the bounce logfile regardless of DSN
     * NOTIFY options, because those options don't apply to postmaster
     * notifications.
     */
    else {
	char   *my_status = mystrdup(my_dsn.status);
	const char *log_status = var_soft_bounce ? "SOFTBOUNCE" : "bounced";

	/*
	 * Supply default action.
	 */
	my_dsn.status = my_status;
	if (var_soft_bounce) {
	    my_status[0] = '4';
	    my_dsn.action = "delayed";
	} else {
	    my_dsn.action = "failed";
	}

	if (mail_command_client(MAIL_CLASS_PRIVATE, var_soft_bounce ?
				var_defer_service : var_bounce_service,
			   SEND_ATTR_INT(MAIL_ATTR_NREQ, BOUNCE_CMD_APPEND),
				SEND_ATTR_INT(MAIL_ATTR_FLAGS, flags),
				SEND_ATTR_STR(MAIL_ATTR_QUEUEID, id),
				SEND_ATTR_FUNC(rcpt_print, (void *) rcpt),
				SEND_ATTR_FUNC(dsn_print, (void *) &my_dsn),
				ATTR_TYPE_END) == 0
	    && ((flags & DEL_REQ_FLAG_RECORD) == 0
		|| trace_append(flags, id, stats, rcpt, relay,
				&my_dsn) == 0)) {
	    log_adhoc(id, stats, rcpt, relay, &my_dsn, log_status);
	    status = (var_soft_bounce ? -1 : 0);
	} else if ((flags & BOUNCE_FLAG_CLEAN) == 0) {
	    VSTRING *junk = vstring_alloc(100);

	    my_dsn.status = "4.3.0";
	    vstring_sprintf(junk, "%s or %s service failure",
			    var_bounce_service, var_trace_service);
	    my_dsn.reason = vstring_str(junk);
	    status = defer_append_intern(flags, id, stats, rcpt, relay, &my_dsn);
	    vstring_free(junk);
	} else {
	    status = -1;
	}
	myfree(my_status);
	return (status);
    }
}

/* bounce_flush - flush the bounce log and deliver to the sender */

int     bounce_flush(int flags, const char *queue, const char *id,
		             const char *encoding, int smtputf8,
		             const char *sender, const char *dsn_envid,
		             int dsn_ret)
{

    /*
     * When we're pretending that we can't bounce, don't send a bounce
     * message.
     */
    if (var_soft_bounce)
	return (-1);
    if (mail_command_client(MAIL_CLASS_PRIVATE, var_bounce_service,
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
    } else if ((flags & BOUNCE_FLAG_CLEAN) == 0) {
	msg_info("%s: status=deferred (bounce failed)", id);
	return (-1);
    } else {
	return (-1);
    }
}

/* bounce_flush_verp - verpified notification */

int     bounce_flush_verp(int flags, const char *queue, const char *id,
			          const char *encoding, int smtputf8,
			          const char *sender, const char *dsn_envid,
			          int dsn_ret, const char *verp_delims)
{

    /*
     * When we're pretending that we can't bounce, don't send a bounce
     * message.
     */
    if (var_soft_bounce)
	return (-1);
    if (mail_command_client(MAIL_CLASS_PRIVATE, var_bounce_service,
			    SEND_ATTR_INT(MAIL_ATTR_NREQ, BOUNCE_CMD_VERP),
			    SEND_ATTR_INT(MAIL_ATTR_FLAGS, flags),
			    SEND_ATTR_STR(MAIL_ATTR_QUEUE, queue),
			    SEND_ATTR_STR(MAIL_ATTR_QUEUEID, id),
			    SEND_ATTR_STR(MAIL_ATTR_ENCODING, encoding),
			    SEND_ATTR_INT(MAIL_ATTR_SMTPUTF8, smtputf8),
			    SEND_ATTR_STR(MAIL_ATTR_SENDER, sender),
			    SEND_ATTR_STR(MAIL_ATTR_DSN_ENVID, dsn_envid),
			    SEND_ATTR_INT(MAIL_ATTR_DSN_RET, dsn_ret),
			    SEND_ATTR_STR(MAIL_ATTR_VERPDL, verp_delims),
			    ATTR_TYPE_END) == 0) {
	return (0);
    } else if ((flags & BOUNCE_FLAG_CLEAN) == 0) {
	msg_info("%s: status=deferred (bounce failed)", id);
	return (-1);
    } else {
	return (-1);
    }
}

/* bounce_one - send notice for one recipient */

int     bounce_one(int flags, const char *queue, const char *id,
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
    if (my_dsn.status[0] != '5' || !dsn_valid(my_dsn.status)) {
	msg_warn("bounce_one: ignoring dsn code \"%s\"", my_dsn.status);
	my_dsn.status = "5.0.0";
    }

    /*
     * DSN filter (Postfix 3.0).
     */
    if (delivery_status_filter != 0
    && (dsn_res = dsn_filter_lookup(delivery_status_filter, &my_dsn)) != 0) {
	if (dsn_res->status[0] == '4')
	    return (defer_append_intern(flags, id, stats, rcpt, relay, dsn_res));
	my_dsn = *dsn_res;
    }
    return (bounce_one_intern(flags, queue, id, encoding, smtputf8, sender,
			  dsn_envid, dsn_ret, stats, rcpt, relay, &my_dsn));
}

/* bounce_one_intern - send notice for one recipient */

int     bounce_one_intern(int flags, const char *queue, const char *id,
			          const char *encoding, int smtputf8,
			          const char *sender, const char *dsn_envid,
			          int dsn_ret, MSG_STATS *stats,
			          RECIPIENT *rcpt, const char *relay,
			          DSN *dsn)
{
    DSN     my_dsn = *dsn;
    int     status;

    /*
     * MTA-requested address verification information is stored in the verify
     * service database.
     */
    if (flags & DEL_REQ_FLAG_MTA_VRFY) {
	my_dsn.action = "undeliverable";
	status = verify_append(id, stats, rcpt, relay, &my_dsn,
			       DEL_RCPT_STAT_BOUNCE);
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
     * When we're not bouncing, then use the standard multi-recipient logfile
     * based procedure.
     */
    else if (var_soft_bounce) {
	return (bounce_append_intern(flags, id, stats, rcpt, relay, &my_dsn));
    }

    /*
     * Normal mail delivery. May also send a delivery record to the user.
     * 
     * XXX DSN We send all recipients regardless of DSN NOTIFY options, because
     * those options don't apply to postmaster notifications.
     */
    else {

	/*
	 * Supply default action.
	 */
	my_dsn.action = "failed";

	if (mail_command_client(MAIL_CLASS_PRIVATE, var_bounce_service,
			      SEND_ATTR_INT(MAIL_ATTR_NREQ, BOUNCE_CMD_ONE),
				SEND_ATTR_INT(MAIL_ATTR_FLAGS, flags),
				SEND_ATTR_STR(MAIL_ATTR_QUEUE, queue),
				SEND_ATTR_STR(MAIL_ATTR_QUEUEID, id),
				SEND_ATTR_STR(MAIL_ATTR_ENCODING, encoding),
				SEND_ATTR_INT(MAIL_ATTR_SMTPUTF8, smtputf8),
				SEND_ATTR_STR(MAIL_ATTR_SENDER, sender),
			      SEND_ATTR_STR(MAIL_ATTR_DSN_ENVID, dsn_envid),
				SEND_ATTR_INT(MAIL_ATTR_DSN_RET, dsn_ret),
				SEND_ATTR_FUNC(rcpt_print, (void *) rcpt),
				SEND_ATTR_FUNC(dsn_print, (void *) &my_dsn),
				ATTR_TYPE_END) == 0
	    && ((flags & DEL_REQ_FLAG_RECORD) == 0
		|| trace_append(flags, id, stats, rcpt, relay,
				&my_dsn) == 0)) {
	    log_adhoc(id, stats, rcpt, relay, &my_dsn, "bounced");
	    status = 0;
	} else if ((flags & BOUNCE_FLAG_CLEAN) == 0) {
	    VSTRING *junk = vstring_alloc(100);

	    my_dsn.status = "4.3.0";
	    vstring_sprintf(junk, "%s or %s service failure",
			    var_bounce_service, var_trace_service);
	    my_dsn.reason = vstring_str(junk);
	    status = defer_append_intern(flags, id, stats, rcpt, relay, &my_dsn);
	    vstring_free(junk);
	} else {
	    status = -1;
	}
	return (status);
    }
}

/* bounce_client_init - initialize bounce/defer DSN filter */

void    bounce_client_init(const char *title, const char *maps)
{
    static const char myname[] = "bounce_client_init";

    if (delivery_status_filter != 0)
	msg_panic("%s: duplicate initialization", myname);
    if (*maps)
	delivery_status_filter = dsn_filter_create(title, maps);
}
