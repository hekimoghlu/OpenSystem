/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 8, 2023.
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
#include <signal.h>
#include <unistd.h>
#include <stdlib.h>

/* Utility library. */

#include <msg.h>
#include <vstring.h>
#include <dict.h>

/* Global library. */

#include <mail_conf.h>
#include <cleanup_user.h>
#include <mail_proto.h>
#include <mail_params.h>
#include <record.h>
#include <rec_type.h>
#include <mail_version.h>

/* Single-threaded server skeleton. */

#include <mail_server.h>

/* Application-specific. */

#include "cleanup.h"

/* cleanup_service - process one request to inject a message into the queue */

static void cleanup_service(VSTREAM *src, char *unused_service, char **argv)
{
    VSTRING *buf = vstring_alloc(100);
    CLEANUP_STATE *state;
    int     flags;
    int     type = 0;
    int     status;

    /*
     * Sanity check. This service takes no command-line arguments.
     */
    if (argv[0])
	msg_fatal("unexpected command-line argument: %s", argv[0]);

    /*
     * Open a queue file and initialize state.
     */
    state = cleanup_open(src);

    /*
     * Send the queue id to the client. Read client processing options. If we
     * can't read the client processing options we can pretty much forget
     * about the whole operation.
     */
    attr_print(src, ATTR_FLAG_NONE,
	       SEND_ATTR_STR(MAIL_ATTR_QUEUEID, state->queue_id),
	       ATTR_TYPE_END);
    if (attr_scan(src, ATTR_FLAG_STRICT,
		  RECV_ATTR_INT(MAIL_ATTR_FLAGS, &flags),
		  ATTR_TYPE_END) != 1) {
	state->errs |= CLEANUP_STAT_BAD;
	flags = 0;
    }
    cleanup_control(state, flags);

    /*
     * XXX Rely on the front-end programs to enforce record size limits.
     * 
     * First, copy the envelope records to the queue file. Then, copy the
     * message content (headers and body). Finally, attach any information
     * extracted from message headers.
     */
    while (CLEANUP_OUT_OK(state)) {
	if ((type = rec_get_raw(src, buf, 0, REC_FLAG_NONE)) < 0) {
	    state->errs |= CLEANUP_STAT_BAD;
	    break;
	}
	if (REC_GET_HIDDEN_TYPE(type)) {
	    msg_warn("%s: record type %d not allowed - discarding this message",
		     state->queue_id, type);
	    state->errs |= CLEANUP_STAT_BAD;
	    break;
	}
	CLEANUP_RECORD(state, type, vstring_str(buf), VSTRING_LEN(buf));
	if (type == REC_TYPE_END)
	    break;
    }

    /*
     * Keep reading in case of problems, until the sender is ready to receive
     * our status report.
     */
    if (CLEANUP_OUT_OK(state) == 0 && type > 0) {
	while (type != REC_TYPE_END
	       && (type = rec_get_raw(src, buf, 0, REC_FLAG_NONE)) > 0) {
	    if (type == REC_TYPE_MILT_COUNT) {
		int     milter_count = atoi(vstring_str(buf));

		/* Avoid deadlock. */
		if (milter_count >= 0)
		    cleanup_milter_receive(state, milter_count);
	    }
	}
    }

    /*
     * Log something to make timeout errors easier to debug.
     */
    if (vstream_ftimeout(src))
	msg_warn("%s: read timeout on %s",
		 state->queue_id, VSTREAM_PATH(src));

    /*
     * Finish this message, and report the result status to the client.
     */
    status = cleanup_flush(state);		/* in case state is modified */
    attr_print(src, ATTR_FLAG_NONE,
	       SEND_ATTR_INT(MAIL_ATTR_STATUS, status),
	       SEND_ATTR_STR(MAIL_ATTR_WHY,
			     (state->flags & CLEANUP_FLAG_SMTP_REPLY)
			     && state->smtp_reply ? state->smtp_reply :
			     state->reason ? state->reason : ""),
	       ATTR_TYPE_END);
    cleanup_free(state);

    /*
     * Cleanup.
     */
    vstring_free(buf);
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

MAIL_VERSION_STAMP_DECLARE;

/* main - the main program */

int     main(int argc, char **argv)
{

    /*
     * Fingerprint executables and core dumps.
     */
    MAIL_VERSION_STAMP_ALLOCATE;

    /*
     * Clean up an incomplete queue file in case of a fatal run-time error,
     * or after receiving SIGTERM from the master at shutdown time.
     */
    signal(SIGTERM, cleanup_sig);
    msg_cleanup(cleanup_all);

    /*
     * Pass control to the single-threaded service skeleton.
     */
    single_server_main(argc, argv, cleanup_service,
		       CA_MAIL_SERVER_INT_TABLE(cleanup_int_table),
		       CA_MAIL_SERVER_BOOL_TABLE(cleanup_bool_table),
		       CA_MAIL_SERVER_STR_TABLE(cleanup_str_table),
		       CA_MAIL_SERVER_TIME_TABLE(cleanup_time_table),
		       CA_MAIL_SERVER_PRE_INIT(cleanup_pre_jail),
		       CA_MAIL_SERVER_POST_INIT(cleanup_post_jail),
		       CA_MAIL_SERVER_PRE_ACCEPT(pre_accept),
		       CA_MAIL_SERVER_IN_FLOW_DELAY,
		       CA_MAIL_SERVER_UNLIMITED,
		       0);
}
