/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 14, 2022.
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
#include <stdlib.h>

/* Utility library. */

#include <msg.h>
#include <vstream.h>

/* Global library. */

#include <deliver_request.h>
#include <mail_queue.h>
#include <bounce.h>
#include <deliver_completed.h>
#include <flush_clnt.h>
#include <sent.h>
#include <dsn_util.h>
#include <mail_version.h>

/* Single server skeleton. */

#include <mail_server.h>

/* deliver_message - deliver message with extreme prejudice */

static int deliver_message(DELIVER_REQUEST *request)
{
    const char *myname = "deliver_message";
    VSTREAM *src;
    int     result = 0;
    int     status;
    RECIPIENT *rcpt;
    int     nrcpt;
    DSN_SPLIT dp;
    DSN     dsn;

    if (msg_verbose)
	msg_info("deliver_message: from %s", request->sender);

    /*
     * Sanity checks.
     */
    if (request->nexthop[0] == 0)
	msg_fatal("empty nexthop hostname");
    if (request->rcpt_list.len <= 0)
	msg_fatal("recipient count: %d", request->rcpt_list.len);

    /*
     * Open the queue file. Opening the file can fail for a variety of
     * reasons, such as the system running out of resources. Instead of
     * throwing away mail, we're raising a fatal error which forces the mail
     * system to back off, and retry later.
     */
    src = mail_queue_open(request->queue_name, request->queue_id,
			  O_RDWR, 0);
    if (src == 0)
	msg_fatal("%s: open %s %s: %m", myname,
		  request->queue_name, request->queue_id);
    if (msg_verbose)
	msg_info("%s: file %s", myname, VSTREAM_PATH(src));

    /*
     * Discard all recipients.
     */
#define BOUNCE_FLAGS(request) DEL_REQ_TRACE_FLAGS(request->flags)

    dsn_split(&dp, "2.0.0", request->nexthop);
    (void) DSN_SIMPLE(&dsn, DSN_STATUS(dp.dsn), dp.text);
    for (nrcpt = 0; nrcpt < request->rcpt_list.len; nrcpt++) {
	rcpt = request->rcpt_list.info + nrcpt;
	status = sent(BOUNCE_FLAGS(request), request->queue_id,
		      &request->msg_stats, rcpt, "none", &dsn);
	if (status == 0 && (request->flags & DEL_REQ_FLAG_SUCCESS))
	    deliver_completed(src, rcpt->offset);
	result |= status;
    }

    /*
     * Clean up.
     */
    if (vstream_fclose(src))
	msg_warn("close %s %s: %m", request->queue_name, request->queue_id);

    return (result);
}

/* discard_service - perform service for client */

static void discard_service(VSTREAM *client_stream, char *unused_service, char **argv)
{
    DELIVER_REQUEST *request;
    int     status;

    /*
     * Sanity check. This service takes no command-line arguments.
     */
    if (argv[0])
	msg_fatal("unexpected command-line argument: %s", argv[0]);

    /*
     * This routine runs whenever a client connects to the UNIX-domain socket
     * dedicated to the discard mailer. What we see below is a little
     * protocol to (1) tell the queue manager that we are ready, (2) read a
     * request from the queue manager, and (3) report the completion status
     * of that request. All connection-management stuff is handled by the
     * common code in single_server.c.
     */
    if ((request = deliver_request_read(client_stream)) != 0) {
	status = deliver_message(request);
	deliver_request_done(client_stream, request, status);
    }
}

/* pre_init - pre-jail initialization */

static void pre_init(char *unused_name, char **unused_argv)
{
    flush_init();
}

MAIL_VERSION_STAMP_DECLARE;

/* main - pass control to the single-threaded skeleton */

int     main(int argc, char **argv)
{

    /*
     * Fingerprint executables and core dumps.
     */
    MAIL_VERSION_STAMP_ALLOCATE;

    single_server_main(argc, argv, discard_service,
		       CA_MAIL_SERVER_PRE_INIT(pre_init),
		       0);
}
