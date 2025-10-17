/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 16, 2024.
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
#include <stdlib.h>
#include <unistd.h>
#include <ctype.h>

/* Utility library. */

#include <msg.h>
#include <events.h>
#include <vstream.h>
#include <dict.h>

/* Global library. */

#include <mail_queue.h>
#include <recipient_list.h>
#include <mail_conf.h>
#include <mail_params.h>
#include <mail_version.h>
#include <mail_proto.h>			/* QMGR_SCAN constants */
#include <mail_flow.h>
#include <flush_clnt.h>

/* Master process interface */

#include <master_proto.h>
#include <mail_server.h>

/* Application-specific. */

#include "qmgr.h"

 /*
  * Tunables.
  */
int     var_queue_run_delay;
int     var_min_backoff_time;
int     var_max_backoff_time;
int     var_max_queue_time;
int     var_dsn_queue_time;
int     var_qmgr_active_limit;
int     var_qmgr_rcpt_limit;
int     var_init_dest_concurrency;
int     var_transport_retry_time;
int     var_dest_con_limit;
int     var_dest_rcpt_limit;
char   *var_defer_xports;
int     var_qmgr_fudge;
int     var_local_rcpt_lim;		/* XXX */
int     var_local_con_lim;		/* XXX */
bool    var_verp_bounce_off;
int     var_qmgr_clog_warn_time;
char   *var_conc_pos_feedback;
char   *var_conc_neg_feedback;
int     var_conc_cohort_limit;
int     var_conc_feedback_debug;
int     var_xport_rate_delay;
int     var_dest_rate_delay;
char   *var_def_filter_nexthop;
int     var_qmgr_daemon_timeout;
int     var_qmgr_ipc_timeout;
int     var_dsn_delay_cleared;
int     var_vrfy_pend_limit;

static QMGR_SCAN *qmgr_scans[2];

#define QMGR_SCAN_IDX_INCOMING 0
#define QMGR_SCAN_IDX_DEFERRED 1
#define QMGR_SCAN_IDX_COUNT (sizeof(qmgr_scans) / sizeof(qmgr_scans[0]))

/* qmgr_deferred_run_event - queue manager heartbeat */

static void qmgr_deferred_run_event(int unused_event, void *dummy)
{

    /*
     * This routine runs when it is time for another deferred queue scan.
     * Make sure this routine gets called again in the future.
     */
    qmgr_scan_request(qmgr_scans[QMGR_SCAN_IDX_DEFERRED], QMGR_SCAN_START);
    event_request_timer(qmgr_deferred_run_event, dummy, var_queue_run_delay);
}

/* qmgr_trigger_event - respond to external trigger(s) */

static void qmgr_trigger_event(char *buf, ssize_t len,
			               char *unused_service, char **argv)
{
    int     incoming_flag = 0;
    int     deferred_flag = 0;
    int     i;

    /*
     * Sanity check. This service takes no command-line arguments.
     */
    if (argv[0])
	msg_fatal("unexpected command-line argument: %s", argv[0]);

    /*
     * Collapse identical requests that have arrived since we looked last
     * time. There is no client feedback so there is no need to process each
     * request in order. And as long as we don't have conflicting requests we
     * are free to sort them into the most suitable order.
     */
#define QMGR_FLUSH_BEFORE	(QMGR_FLUSH_ONCE | QMGR_FLUSH_DFXP)

    for (i = 0; i < len; i++) {
	if (msg_verbose)
	    msg_info("request: %d (%c)",
		     buf[i], ISALNUM(buf[i]) ? buf[i] : '?');
	switch (buf[i]) {
	case TRIGGER_REQ_WAKEUP:
	case QMGR_REQ_SCAN_INCOMING:
	    incoming_flag |= QMGR_SCAN_START;
	    break;
	case QMGR_REQ_SCAN_DEFERRED:
	    deferred_flag |= QMGR_SCAN_START;
	    break;
	case QMGR_REQ_FLUSH_DEAD:
	    deferred_flag |= QMGR_FLUSH_BEFORE;
	    incoming_flag |= QMGR_FLUSH_BEFORE;
	    break;
	case QMGR_REQ_SCAN_ALL:
	    deferred_flag |= QMGR_SCAN_ALL;
	    incoming_flag |= QMGR_SCAN_ALL;
	    break;
	default:
	    if (msg_verbose)
		msg_info("request ignored");
	    break;
	}
    }

    /*
     * Process each request type at most once. Modifiers take effect upon the
     * next queue run. If no queue run is in progress, and a queue scan is
     * requested, the request takes effect immediately.
     */
    if (incoming_flag != 0)
	qmgr_scan_request(qmgr_scans[QMGR_SCAN_IDX_INCOMING], incoming_flag);
    if (deferred_flag != 0)
	qmgr_scan_request(qmgr_scans[QMGR_SCAN_IDX_DEFERRED], deferred_flag);
}

/* qmgr_loop - queue manager main loop */

static int qmgr_loop(char *unused_name, char **unused_argv)
{
    char   *path;
    ssize_t token_count;
    int     feed = 0;
    int     scan_idx;			/* Priority order scan index */
    static int first_scan_idx = QMGR_SCAN_IDX_INCOMING;
    int     last_scan_idx = QMGR_SCAN_IDX_COUNT - 1;
    int     delay;

    /*
     * This routine runs as part of the event handling loop, after the event
     * manager has delivered a timer or I/O event (including the completion
     * of a connection to a delivery process), or after it has waited for a
     * specified amount of time. The result value of qmgr_loop() specifies
     * how long the event manager should wait for the next event.
     */
#define DONT_WAIT	0
#define WAIT_FOR_EVENT	(-1)

    /*
     * Attempt to drain the active queue by allocating a suitable delivery
     * process and by delivering mail via it. Delivery process allocation and
     * mail delivery are asynchronous.
     */
    qmgr_active_drain();

    /*
     * Let some new blood into the active queue when the queue size is
     * smaller than some configurable limit, and when the number of in-core
     * recipients does not exceed some configurable limit.
     * 
     * We import one message per interrupt, to optimally tune the input count
     * for the number of delivery agent protocol wait states, as explained in
     * qmgr_transport.c.
     */
    delay = WAIT_FOR_EVENT;
    for (scan_idx = 0; qmgr_message_count < var_qmgr_active_limit
	 && qmgr_recipient_count < var_qmgr_rcpt_limit
	 && scan_idx < QMGR_SCAN_IDX_COUNT; ++scan_idx) {
	last_scan_idx = (scan_idx + first_scan_idx) % QMGR_SCAN_IDX_COUNT;
	if ((path = qmgr_scan_next(qmgr_scans[last_scan_idx])) != 0) {
	    delay = DONT_WAIT;
	    if ((feed = qmgr_active_feed(qmgr_scans[last_scan_idx], path)) != 0)
		break;
	}
    }

    /*
     * Round-robin the queue scans. When the active queue becomes full,
     * prefer new mail over deferred mail.
     */
    if (qmgr_message_count < var_qmgr_active_limit
	&& qmgr_recipient_count < var_qmgr_rcpt_limit) {
	first_scan_idx = (last_scan_idx + 1) % QMGR_SCAN_IDX_COUNT;
    } else if (first_scan_idx != QMGR_SCAN_IDX_INCOMING) {
	first_scan_idx = QMGR_SCAN_IDX_INCOMING;
    }

    /*
     * Global flow control. If enabled, slow down receiving processes that
     * get ahead of the queue manager, but don't block them completely.
     */
    if (var_in_flow_delay > 0) {
	token_count = mail_flow_count();
	if (token_count < var_proc_limit) {
	    if (feed != 0 && last_scan_idx == QMGR_SCAN_IDX_INCOMING)
		mail_flow_put(1);
	    else if (qmgr_scans[QMGR_SCAN_IDX_INCOMING]->handle == 0)
		mail_flow_put(var_proc_limit - token_count);
	} else if (token_count > var_proc_limit) {
	    mail_flow_get(token_count - var_proc_limit);
	}
    }
    return (delay);
}

/* pre_accept - see if tables have changed */

static void pre_accept(char *unused_name, char **unused_argv)
{
    const char *table;

    if ((table = dict_changed_name()) != 0) {
	msg_info("table %s has changed -- restarting", table);
	exit(0);
    }
}

/* qmgr_pre_init - pre-jail initialization */

static void qmgr_pre_init(char *unused_name, char **unused_argv)
{
    flush_init();
}

/* qmgr_post_init - post-jail initialization */

static void qmgr_post_init(char *unused_name, char **unused_argv)
{

    /*
     * Sanity check.
     */
    if (var_qmgr_rcpt_limit < var_qmgr_active_limit) {
	msg_warn("%s is smaller than %s - adjusting %s",
	      VAR_QMGR_RCPT_LIMIT, VAR_QMGR_ACT_LIMIT, VAR_QMGR_RCPT_LIMIT);
	var_qmgr_rcpt_limit = var_qmgr_active_limit;
    }
    if (var_dsn_queue_time > var_max_queue_time) {
	msg_warn("%s is larger than %s - adjusting %s",
		 VAR_DSN_QUEUE_TIME, VAR_MAX_QUEUE_TIME, VAR_DSN_QUEUE_TIME);
	var_dsn_queue_time = var_max_queue_time;
    }

    /*
     * This routine runs after the skeleton code has entered the chroot jail.
     * Prevent automatic process suicide after a limited number of client
     * requests or after a limited amount of idle time. Move any left-over
     * entries from the active queue to the incoming queue, and give them a
     * time stamp into the future, in order to allow ongoing deliveries to
     * finish first. Start scanning the incoming and deferred queues.
     * Left-over active queue entries are moved to the incoming queue because
     * the incoming queue has priority; moving left-overs to the deferred
     * queue could cause anomalous delays when "postfix reload/start" are
     * issued often. Override the IPC timeout (default 3600s) so that the
     * queue manager can reset a broken IPC channel before the watchdog timer
     * goes off.
     */
    var_ipc_timeout = var_qmgr_ipc_timeout;
    var_use_limit = 0;
    var_idle_limit = 0;
    qmgr_move(MAIL_QUEUE_ACTIVE, MAIL_QUEUE_INCOMING, event_time());
    qmgr_scans[QMGR_SCAN_IDX_INCOMING] = qmgr_scan_create(MAIL_QUEUE_INCOMING);
    qmgr_scans[QMGR_SCAN_IDX_DEFERRED] = qmgr_scan_create(MAIL_QUEUE_DEFERRED);
    qmgr_scan_request(qmgr_scans[QMGR_SCAN_IDX_INCOMING], QMGR_SCAN_START);
    qmgr_deferred_run_event(0, (void *) 0);
}

MAIL_VERSION_STAMP_DECLARE;

/* main - the main program */

int     main(int argc, char **argv)
{
    static const CONFIG_STR_TABLE str_table[] = {
	VAR_DEFER_XPORTS, DEF_DEFER_XPORTS, &var_defer_xports, 0, 0,
	VAR_CONC_POS_FDBACK, DEF_CONC_POS_FDBACK, &var_conc_pos_feedback, 1, 0,
	VAR_CONC_NEG_FDBACK, DEF_CONC_NEG_FDBACK, &var_conc_neg_feedback, 1, 0,
	VAR_DEF_FILTER_NEXTHOP, DEF_DEF_FILTER_NEXTHOP, &var_def_filter_nexthop, 0, 0,
	0,
    };
    static const CONFIG_TIME_TABLE time_table[] = {
	VAR_QUEUE_RUN_DELAY, DEF_QUEUE_RUN_DELAY, &var_queue_run_delay, 1, 0,
	VAR_MIN_BACKOFF_TIME, DEF_MIN_BACKOFF_TIME, &var_min_backoff_time, 1, 0,
	VAR_MAX_BACKOFF_TIME, DEF_MAX_BACKOFF_TIME, &var_max_backoff_time, 1, 0,
	VAR_MAX_QUEUE_TIME, DEF_MAX_QUEUE_TIME, &var_max_queue_time, 0, 8640000,
	VAR_DSN_QUEUE_TIME, DEF_DSN_QUEUE_TIME, &var_dsn_queue_time, 0, 8640000,
	VAR_XPORT_RETRY_TIME, DEF_XPORT_RETRY_TIME, &var_transport_retry_time, 1, 0,
	VAR_QMGR_CLOG_WARN_TIME, DEF_QMGR_CLOG_WARN_TIME, &var_qmgr_clog_warn_time, 0, 0,
	VAR_XPORT_RATE_DELAY, DEF_XPORT_RATE_DELAY, &var_xport_rate_delay, 0, 0,
	VAR_DEST_RATE_DELAY, DEF_DEST_RATE_DELAY, &var_dest_rate_delay, 0, 0,
	VAR_QMGR_DAEMON_TIMEOUT, DEF_QMGR_DAEMON_TIMEOUT, &var_qmgr_daemon_timeout, 1, 0,
	VAR_QMGR_IPC_TIMEOUT, DEF_QMGR_IPC_TIMEOUT, &var_qmgr_ipc_timeout, 1, 0,
	0,
    };
    static const CONFIG_INT_TABLE int_table[] = {
	VAR_QMGR_ACT_LIMIT, DEF_QMGR_ACT_LIMIT, &var_qmgr_active_limit, 1, 0,
	VAR_QMGR_RCPT_LIMIT, DEF_QMGR_RCPT_LIMIT, &var_qmgr_rcpt_limit, 1, 0,
	VAR_INIT_DEST_CON, DEF_INIT_DEST_CON, &var_init_dest_concurrency, 1, 0,
	VAR_DEST_CON_LIMIT, DEF_DEST_CON_LIMIT, &var_dest_con_limit, 0, 0,
	VAR_DEST_RCPT_LIMIT, DEF_DEST_RCPT_LIMIT, &var_dest_rcpt_limit, 0, 0,
	VAR_QMGR_FUDGE, DEF_QMGR_FUDGE, &var_qmgr_fudge, 10, 100,
	VAR_LOCAL_RCPT_LIMIT, DEF_LOCAL_RCPT_LIMIT, &var_local_rcpt_lim, 0, 0,
	VAR_LOCAL_CON_LIMIT, DEF_LOCAL_CON_LIMIT, &var_local_con_lim, 0, 0,
	VAR_CONC_COHORT_LIM, DEF_CONC_COHORT_LIM, &var_conc_cohort_limit, 0, 0,
	VAR_VRFY_PEND_LIMIT, DEF_VRFY_PEND_LIMIT, &var_vrfy_pend_limit, 1, 0,
	0,
    };
    static const CONFIG_BOOL_TABLE bool_table[] = {
	VAR_VERP_BOUNCE_OFF, DEF_VERP_BOUNCE_OFF, &var_verp_bounce_off,
	VAR_CONC_FDBACK_DEBUG, DEF_CONC_FDBACK_DEBUG, &var_conc_feedback_debug,
	VAR_DSN_DELAY_CLEARED, DEF_DSN_DELAY_CLEARED, &var_dsn_delay_cleared,
	0,
    };

    /*
     * Fingerprint executables and core dumps.
     */
    MAIL_VERSION_STAMP_ALLOCATE;

    /*
     * Use the trigger service skeleton, because no-one else should be
     * monitoring our service port while this process runs, and because we do
     * not talk back to the client.
     */
    trigger_server_main(argc, argv, qmgr_trigger_event,
			CA_MAIL_SERVER_INT_TABLE(int_table),
			CA_MAIL_SERVER_STR_TABLE(str_table),
			CA_MAIL_SERVER_BOOL_TABLE(bool_table),
			CA_MAIL_SERVER_TIME_TABLE(time_table),
			CA_MAIL_SERVER_PRE_INIT(qmgr_pre_init),
			CA_MAIL_SERVER_POST_INIT(qmgr_post_init),
			CA_MAIL_SERVER_LOOP(qmgr_loop),
			CA_MAIL_SERVER_PRE_ACCEPT(pre_accept),
			CA_MAIL_SERVER_SOLITARY,
			CA_MAIL_SERVER_WATCHDOG(&var_qmgr_daemon_timeout),
			0);
}
