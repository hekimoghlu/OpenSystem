/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 14, 2022.
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
#include <stdlib.h>			/* 44BSD stdarg.h uses abort() */
#include <stdarg.h>
#include <string.h>

/* Utility library. */

#include <msg.h>
#include <vstring.h>
#include <stringops.h>

/* Global library. */

#include <smtp_stream.h>
#include <deliver_request.h>
#include <deliver_completed.h>
#include <bounce.h>
#include <defer.h>
#include <mail_error.h>
#include <dsn_buf.h>
#include <dsn.h>
#include <mail_params.h>

/* Application-specific. */

#include "smtp.h"
#include "smtp_sasl.h"

/* smtp_check_code - check response code */

static void smtp_check_code(SMTP_SESSION *session, int code)
{

    /*
     * The intention of this code is to alert the postmaster when the local
     * Postfix SMTP client screws up, protocol wise. RFC 821 says that x0z
     * replies "refer to syntax errors, syntactically correct commands that
     * don't fit any functional category, and unimplemented or superfluous
     * commands". Unfortunately, this also triggers postmaster notices when
     * remote servers screw up, protocol wise. This is becoming a common
     * problem now that response codes are configured manually as part of
     * anti-UCE systems, by people who aren't aware of RFC details.
     */
    if (code < 400 || code > 599
	|| code == 555			/* RFC 1869, section 6.1. */
	|| (code >= 500 && code < 510))
	session->error_mask |= MAIL_ERROR_PROTOCOL;
}

/* smtp_bulk_fail - skip, defer or bounce recipients, maybe throttle queue */

static int smtp_bulk_fail(SMTP_STATE *state, int throttle_queue)
{
    DELIVER_REQUEST *request = state->request;
    SMTP_SESSION *session = state->session;
    DSN_BUF *why = state->why;
    RECIPIENT *rcpt;
    int     status;
    int     aggregate_status;
    int     soft_error = (STR(why->status)[0] == '4');
    int     soft_bounce_error = (STR(why->status)[0] == '5' && var_soft_bounce);
    int     nrcpt;

    /*
     * Don't defer the recipients just yet when this error qualifies them for
     * delivery to a backup server. Just log something informative to show
     * why we're skipping this host.
     */
    if ((soft_error || soft_bounce_error)
	&& (state->misc_flags & SMTP_MISC_FLAG_FINAL_SERVER) == 0) {
	msg_info("%s: %s", request->queue_id, STR(why->reason));
	for (nrcpt = 0; nrcpt < SMTP_RCPT_LEFT(state); nrcpt++) {
	    rcpt = request->rcpt_list.info + nrcpt;
	    if (SMTP_RCPT_ISMARKED(rcpt))
		continue;
	    SMTP_RCPT_KEEP(state, rcpt);
	}
    }

    /*
     * Defer or bounce all the remaining recipients, and delete them from the
     * delivery request. If a bounce fails, defer instead and do not qualify
     * the recipient for delivery to a backup server.
     */
    else {

	/*
	 * If we are still in the connection set-up phase, update the set-up
	 * completion time here, otherwise the time spent in set-up latency
	 * will be attributed as message transfer latency.
	 * 
	 * All remaining recipients have failed at this point, so we update the
	 * delivery completion time stamp so that multiple recipient status
	 * records show the same delay values.
	 */
	if (request->msg_stats.conn_setup_done.tv_sec == 0) {
	    GETTIMEOFDAY(&request->msg_stats.conn_setup_done);
	    request->msg_stats.deliver_done =
		request->msg_stats.conn_setup_done;
	} else
	    GETTIMEOFDAY(&request->msg_stats.deliver_done);

	(void) DSN_FROM_DSN_BUF(why);
	aggregate_status = 0;
	for (nrcpt = 0; nrcpt < SMTP_RCPT_LEFT(state); nrcpt++) {
	    rcpt = request->rcpt_list.info + nrcpt;
	    if (SMTP_RCPT_ISMARKED(rcpt))
		continue;
	    status = (soft_error ? defer_append : bounce_append)
		(DEL_REQ_TRACE_FLAGS(request->flags), request->queue_id,
		 &request->msg_stats, rcpt,
		 session ? session->namaddrport : "none", &why->dsn);
	    if (status == 0)
		deliver_completed(state->src, rcpt->offset);
	    SMTP_RCPT_DROP(state, rcpt);
	    aggregate_status |= status;
	}
	state->status |= aggregate_status;
	if ((state->misc_flags & SMTP_MISC_FLAG_COMPLETE_SESSION) == 0
	    && throttle_queue && aggregate_status
	    && request->hop_status == 0)
	    request->hop_status = DSN_COPY(&why->dsn);
    }

    /*
     * Don't cache this session. We can't talk to this server.
     */
    if (throttle_queue && session)
	DONT_CACHE_THROTTLED_SESSION;

    return (-1);
}

/* smtp_sess_fail - skip site, defer or bounce all recipients */

int     smtp_sess_fail(SMTP_STATE *state)
{

    /*
     * We can't avoid copying copying lots of strings into VSTRING buffers,
     * because this error information is collected by a routine that
     * terminates BEFORE the error is reported.
     */
    return (smtp_bulk_fail(state, SMTP_THROTTLE));
}

/* vsmtp_fill_dsn - fill in temporary DSN structure */

static void vsmtp_fill_dsn(SMTP_STATE *state, const char *mta_name,
			           const char *status, const char *reply,
			           const char *format, va_list ap)
{
    DSN_BUF *why = state->why;

    /*
     * We could avoid copying lots of strings into VSTRING buffers, because
     * this error information is given to us by a routine that terminates
     * AFTER the error is reported. However, this results in ugly kludges
     * when informal text needs to be formatted. So we maintain consistency
     * with other error reporting in the SMTP client even if we waste a few
     * cycles.
     */
    VSTRING_RESET(why->reason);
    if (mta_name && status && status[0] != '4' && status[0] != '5') {
	vstring_strcpy(why->reason, "Protocol error: ");
	status = "5.5.0";
    }
    vstring_vsprintf_append(why->reason, format, ap);
    dsb_formal(why, status, DSB_DEF_ACTION,
	       mta_name ? DSB_MTYPE_DNS : DSB_MTYPE_NONE, mta_name,
	       reply ? DSB_DTYPE_SMTP : DSB_DTYPE_NONE, reply);
}

/* smtp_misc_fail - maybe throttle queue; skip/defer/bounce all recipients */

int     smtp_misc_fail(SMTP_STATE *state, int throttle, const char *mta_name, 
				SMTP_RESP *resp, const char *format,...)
{
    va_list ap;

    /*
     * Initialize.
     */
    va_start(ap, format);
    vsmtp_fill_dsn(state, mta_name, resp->dsn, resp->str, format, ap);
    va_end(ap);

    if (state->session && mta_name)
	smtp_check_code(state->session, resp->code);

    /*
     * Skip, defer or bounce recipients, and throttle this queue.
     */
    return (smtp_bulk_fail(state, throttle));
}

/* smtp_rcpt_fail - skip, defer, or bounce recipient */

void    smtp_rcpt_fail(SMTP_STATE *state, RECIPIENT *rcpt, const char *mta_name,
		               SMTP_RESP *resp, const char *format,...)
{
    DELIVER_REQUEST *request = state->request;
    SMTP_SESSION *session = state->session;
    DSN_BUF *why = state->why;
    int     status;
    int     soft_error;
    int     soft_bounce_error;
    va_list ap;

    /*
     * Sanity check.
     */
    if (SMTP_RCPT_ISMARKED(rcpt))
	msg_panic("smtp_rcpt_fail: recipient <%s> is marked", rcpt->address);

    /*
     * Initialize.
     */
    va_start(ap, format);
    vsmtp_fill_dsn(state, mta_name, resp->dsn, resp->str, format, ap);
    va_end(ap);
    soft_error = STR(why->status)[0] == '4';
    soft_bounce_error = (STR(why->status)[0] == '5' && var_soft_bounce);

    if (state->session && mta_name)
	smtp_check_code(state->session, resp->code);

    /*
     * Don't defer this recipient record just yet when this error qualifies
     * for trying other mail servers. Just log something informative to show
     * why we're skipping this recipient now.
     */
    if ((soft_error || soft_bounce_error)
	&& (state->misc_flags & SMTP_MISC_FLAG_FINAL_SERVER) == 0) {
	msg_info("%s: %s", request->queue_id, STR(why->reason));
	SMTP_RCPT_KEEP(state, rcpt);
    }

    /*
     * Defer or bounce this recipient, and delete from the delivery request.
     * If the bounce fails, defer instead and do not qualify the recipient
     * for delivery to a backup server.
     * 
     * Note: we may still make an SMTP connection to deliver other recipients
     * that did qualify for delivery to a backup server.
     */
    else {
	(void) DSN_FROM_DSN_BUF(state->why);
	status = (soft_error ? defer_append : bounce_append)
	    (DEL_REQ_TRACE_FLAGS(request->flags), request->queue_id,
	     &request->msg_stats, rcpt,
	     session ? session->namaddrport : "none", &why->dsn);
	if (status == 0)
	    deliver_completed(state->src, rcpt->offset);
	SMTP_RCPT_DROP(state, rcpt);
	state->status |= status;
    }
}

/* smtp_stream_except - defer domain after I/O problem */

int     smtp_stream_except(SMTP_STATE *state, int code, const char *description)
{
    SMTP_SESSION *session = state->session;
    DSN_BUF *why = state->why;

    /*
     * Sanity check.
     */
    if (session == 0)
	msg_panic("smtp_stream_except: no session");

    /*
     * Initialize.
     */
    switch (code) {
    default:
	msg_panic("smtp_stream_except: unknown exception %d", code);
    case SMTP_ERR_EOF:
	dsb_simple(why, "4.4.2", "lost connection with %s while %s",
		   session->namaddr, description);
#ifdef USE_TLS
	if (PLAINTEXT_FALLBACK_OK_AFTER_TLS_SESSION_FAILURE)
	    RETRY_AS_PLAINTEXT;
#endif
	break;
    case SMTP_ERR_TIME:
	dsb_simple(why, "4.4.2", "conversation with %s timed out while %s",
		   session->namaddr, description);
#ifdef USE_TLS
	if (PLAINTEXT_FALLBACK_OK_AFTER_TLS_SESSION_FAILURE)
	    RETRY_AS_PLAINTEXT;
#endif
	break;
    case SMTP_ERR_DATA:
	session->error_mask |= MAIL_ERROR_DATA;
	dsb_simple(why, "4.3.0", "local data error while talking to %s",
		   session->namaddr);
    }

    /*
     * The smtp_bulk_fail() call below will not throttle the destination when
     * falling back to plaintext, because RETRY_AS_PLAINTEXT clears the
     * FINAL_SERVER flag.
     */
    return (smtp_bulk_fail(state, SMTP_THROTTLE));
}
