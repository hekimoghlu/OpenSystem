/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 17, 2022.
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
#include <sys/stat.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>

/* Utility library. */

#include <msg.h>
#include <vstream.h>
#include <vstring.h>
#include <mymalloc.h>
#include <iostuff.h>
#include <myflock.h>

/* Global library. */

#include "mail_queue.h"
#include "mail_proto.h"
#include "mail_open_ok.h"
#include "recipient_list.h"
#include "dsn.h"
#include "dsn_print.h"
#include "deliver_request.h"
#include "rcpt_buf.h"

/* deliver_request_initial - send initial status code */

static int deliver_request_initial(VSTREAM *stream)
{
    int     err;

    /*
     * The master processes runs a finite number of delivery agent processes
     * to handle service requests. Thus, a delivery agent process must send
     * something to inform the queue manager that it is ready to receive a
     * delivery request; otherwise the queue manager could block in write().
     */
    if (msg_verbose)
	msg_info("deliver_request_initial: send initial status");
    attr_print(stream, ATTR_FLAG_NONE,
	       SEND_ATTR_INT(MAIL_ATTR_STATUS, 0),
	       ATTR_TYPE_END);
    if ((err = vstream_fflush(stream)) != 0)
	if (msg_verbose)
	    msg_warn("send initial status: %m");
    return (err);
}

/* deliver_request_final - send final delivery request status */

static int deliver_request_final(VSTREAM *stream, DELIVER_REQUEST *request,
				         int status)
{
    DSN    *hop_status;
    int     err;

    /* XXX This DSN structure initialization bypasses integrity checks. */
    static DSN dummy_dsn = {"", "", "", "", "", "", ""};

    /*
     * Send the status and the optional reason.
     */
    if ((hop_status = request->hop_status) == 0)
	hop_status = &dummy_dsn;
    if (msg_verbose)
	msg_info("deliver_request_final: send: \"%s\" %d",
		 hop_status->reason, status);
    attr_print(stream, ATTR_FLAG_NONE,
	       SEND_ATTR_FUNC(dsn_print, (void *) hop_status),
	       SEND_ATTR_INT(MAIL_ATTR_STATUS, status),
	       ATTR_TYPE_END);
    if ((err = vstream_fflush(stream)) != 0)
	if (msg_verbose)
	    msg_warn("send final status: %m");

    /*
     * With some UNIX systems, stream sockets lose data when you close them
     * immediately after writing to them. That is not how sockets are
     * supposed to behave! The workaround is to wait until the receiver
     * closes the connection. Calling VSTREAM_GETC() has the benefit of using
     * whatever timeout is specified in the ipc_timeout parameter.
     */
    (void) VSTREAM_GETC(stream);
    return (err);
}

/* deliver_request_get - receive message delivery request */

static int deliver_request_get(VSTREAM *stream, DELIVER_REQUEST *request)
{
    const char *myname = "deliver_request_get";
    const char *path;
    struct stat st;
    static VSTRING *queue_name;
    static VSTRING *queue_id;
    static VSTRING *nexthop;
    static VSTRING *encoding;
    static VSTRING *address;
    static VSTRING *client_name;
    static VSTRING *client_addr;
    static VSTRING *client_port;
    static VSTRING *client_proto;
    static VSTRING *client_helo;
    static VSTRING *sasl_method;
    static VSTRING *sasl_username;
    static VSTRING *sasl_sender;
    static VSTRING *log_ident;
    static VSTRING *rewrite_context;
    static VSTRING *dsn_envid;
    static RCPT_BUF *rcpt_buf;
    int     rcpt_count;
    int     smtputf8;
    int     dsn_ret;

    /*
     * Initialize. For some reason I wanted to allow for multiple instances
     * of a deliver_request structure, thus the hoopla with string
     * initialization and copying.
     */
    if (queue_name == 0) {
	queue_name = vstring_alloc(10);
	queue_id = vstring_alloc(10);
	nexthop = vstring_alloc(10);
	encoding = vstring_alloc(10);
	address = vstring_alloc(10);
	client_name = vstring_alloc(10);
	client_addr = vstring_alloc(10);
	client_port = vstring_alloc(10);
	client_proto = vstring_alloc(10);
	client_helo = vstring_alloc(10);
	sasl_method = vstring_alloc(10);
	sasl_username = vstring_alloc(10);
	sasl_sender = vstring_alloc(10);
	log_ident = vstring_alloc(10);
	rewrite_context = vstring_alloc(10);
	dsn_envid = vstring_alloc(10);
	rcpt_buf = rcpb_create();
    }

    /*
     * Extract the queue file name, data offset, and sender address. Abort
     * the conversation when they send bad information.
     */
    if (attr_scan(stream, ATTR_FLAG_STRICT,
		  RECV_ATTR_INT(MAIL_ATTR_FLAGS, &request->flags),
		  RECV_ATTR_STR(MAIL_ATTR_QUEUE, queue_name),
		  RECV_ATTR_STR(MAIL_ATTR_QUEUEID, queue_id),
		  RECV_ATTR_LONG(MAIL_ATTR_OFFSET, &request->data_offset),
		  RECV_ATTR_LONG(MAIL_ATTR_SIZE, &request->data_size),
		  RECV_ATTR_STR(MAIL_ATTR_NEXTHOP, nexthop),
		  RECV_ATTR_STR(MAIL_ATTR_ENCODING, encoding),
		  RECV_ATTR_INT(MAIL_ATTR_SMTPUTF8, &smtputf8),
		  RECV_ATTR_STR(MAIL_ATTR_SENDER, address),
		  RECV_ATTR_STR(MAIL_ATTR_DSN_ENVID, dsn_envid),
		  RECV_ATTR_INT(MAIL_ATTR_DSN_RET, &dsn_ret),
	       RECV_ATTR_FUNC(msg_stats_scan, (void *) &request->msg_stats),
    /* XXX Should be encapsulated with ATTR_TYPE_FUNC. */
		  RECV_ATTR_STR(MAIL_ATTR_LOG_CLIENT_NAME, client_name),
		  RECV_ATTR_STR(MAIL_ATTR_LOG_CLIENT_ADDR, client_addr),
		  RECV_ATTR_STR(MAIL_ATTR_LOG_CLIENT_PORT, client_port),
		  RECV_ATTR_STR(MAIL_ATTR_LOG_PROTO_NAME, client_proto),
		  RECV_ATTR_STR(MAIL_ATTR_LOG_HELO_NAME, client_helo),
    /* XXX Should be encapsulated with ATTR_TYPE_FUNC. */
		  RECV_ATTR_STR(MAIL_ATTR_SASL_METHOD, sasl_method),
		  RECV_ATTR_STR(MAIL_ATTR_SASL_USERNAME, sasl_username),
		  RECV_ATTR_STR(MAIL_ATTR_SASL_SENDER, sasl_sender),
    /* XXX Ditto if we want to pass TLS certificate info. */
		  RECV_ATTR_STR(MAIL_ATTR_LOG_IDENT, log_ident),
		  RECV_ATTR_STR(MAIL_ATTR_RWR_CONTEXT, rewrite_context),
		  RECV_ATTR_INT(MAIL_ATTR_RCPT_COUNT, &rcpt_count),
		  ATTR_TYPE_END) != 23) {
	msg_warn("%s: error receiving common attributes", myname);
	return (-1);
    }
    if (mail_open_ok(vstring_str(queue_name),
		     vstring_str(queue_id), &st, &path) == 0)
	return (-1);

    /* Don't override hand-off time after deliver_pass() delegation. */
    if (request->msg_stats.agent_handoff.tv_sec == 0)
	GETTIMEOFDAY(&request->msg_stats.agent_handoff);

    request->queue_name = mystrdup(vstring_str(queue_name));
    request->queue_id = mystrdup(vstring_str(queue_id));
    request->nexthop = mystrdup(vstring_str(nexthop));
    request->encoding = mystrdup(vstring_str(encoding));
    /* Fix 20140708: dedicated smtputf8 attribute with its own flags. */
    request->smtputf8 = smtputf8;
    request->sender = mystrdup(vstring_str(address));
    request->client_name = mystrdup(vstring_str(client_name));
    request->client_addr = mystrdup(vstring_str(client_addr));
    request->client_port = mystrdup(vstring_str(client_port));
    request->client_proto = mystrdup(vstring_str(client_proto));
    request->client_helo = mystrdup(vstring_str(client_helo));
    request->sasl_method = mystrdup(vstring_str(sasl_method));
    request->sasl_username = mystrdup(vstring_str(sasl_username));
    request->sasl_sender = mystrdup(vstring_str(sasl_sender));
    request->log_ident = mystrdup(vstring_str(log_ident));
    request->rewrite_context = mystrdup(vstring_str(rewrite_context));
    request->dsn_envid = mystrdup(vstring_str(dsn_envid));
    request->dsn_ret = dsn_ret;

    /*
     * Extract the recipient offset and address list. Skip over any
     * attributes from the sender that we do not understand.
     */
    while (rcpt_count-- > 0) {
	if (attr_scan(stream, ATTR_FLAG_STRICT,
		      RECV_ATTR_FUNC(rcpb_scan, (void *) rcpt_buf),
		      ATTR_TYPE_END) != 1) {
	    msg_warn("%s: error receiving recipient attributes", myname);
	    return (-1);
	}
	recipient_list_add(&request->rcpt_list, rcpt_buf->offset,
			   vstring_str(rcpt_buf->dsn_orcpt),
			   rcpt_buf->dsn_notify,
			   vstring_str(rcpt_buf->orig_addr),
			   vstring_str(rcpt_buf->address));
    }
    if (request->rcpt_list.len <= 0) {
	msg_warn("%s: no recipients in delivery request for destination %s",
		 request->queue_id, request->nexthop);
	return (-1);
    }

    /*
     * Open the queue file and set a shared lock, in order to prevent
     * duplicate deliveries when the queue is flushed immediately after queue
     * manager restart.
     * 
     * The queue manager locks the file exclusively when it enters the active
     * queue, and releases the lock before starting deliveries from that
     * file. The queue manager does not lock the file again when reading more
     * recipients into memory. When the queue manager is restarted, the new
     * process moves files from the active queue to the incoming queue to
     * cool off for a while. Delivery agents should therefore never try to
     * open a file that is locked by a queue manager process.
     * 
     * Opening the queue file can fail for a variety of reasons, such as the
     * system running out of resources. Instead of throwing away mail, we're
     * raising a fatal error which forces the mail system to back off, and
     * retry later.
     */
#define DELIVER_LOCK_MODE (MYFLOCK_OP_SHARED | MYFLOCK_OP_NOWAIT)

    request->fp =
	mail_queue_open(request->queue_name, request->queue_id, O_RDWR, 0);
    if (request->fp == 0) {
	if (errno != ENOENT)
	    msg_fatal("open %s %s: %m", request->queue_name, request->queue_id);
	msg_warn("open %s %s: %m", request->queue_name, request->queue_id);
	return (-1);
    }
    if (msg_verbose)
	msg_info("%s: file %s", myname, VSTREAM_PATH(request->fp));
    if (myflock(vstream_fileno(request->fp), INTERNAL_LOCK, DELIVER_LOCK_MODE) < 0)
	msg_fatal("shared lock %s: %m", VSTREAM_PATH(request->fp));
    close_on_exec(vstream_fileno(request->fp), CLOSE_ON_EXEC);

    return (0);
}

/* deliver_request_alloc - allocate delivery request structure */

static DELIVER_REQUEST *deliver_request_alloc(void)
{
    DELIVER_REQUEST *request;

    request = (DELIVER_REQUEST *) mymalloc(sizeof(*request));
    request->fp = 0;
    request->queue_name = 0;
    request->queue_id = 0;
    request->nexthop = 0;
    request->encoding = 0;
    request->sender = 0;
    request->data_offset = 0;
    request->data_size = 0;
    recipient_list_init(&request->rcpt_list, RCPT_LIST_INIT_STATUS);
    request->hop_status = 0;
    request->client_name = 0;
    request->client_addr = 0;
    request->client_port = 0;
    request->client_proto = 0;
    request->client_helo = 0;
    request->sasl_method = 0;
    request->sasl_username = 0;
    request->sasl_sender = 0;
    request->log_ident = 0;
    request->rewrite_context = 0;
    request->dsn_envid = 0;
    return (request);
}

/* deliver_request_free - clean up delivery request structure */

static void deliver_request_free(DELIVER_REQUEST *request)
{
    if (request->fp)
	vstream_fclose(request->fp);
    if (request->queue_name)
	myfree(request->queue_name);
    if (request->queue_id)
	myfree(request->queue_id);
    if (request->nexthop)
	myfree(request->nexthop);
    if (request->encoding)
	myfree(request->encoding);
    if (request->sender)
	myfree(request->sender);
    recipient_list_free(&request->rcpt_list);
    if (request->hop_status)
	dsn_free(request->hop_status);
    if (request->client_name)
	myfree(request->client_name);
    if (request->client_addr)
	myfree(request->client_addr);
    if (request->client_port)
	myfree(request->client_port);
    if (request->client_proto)
	myfree(request->client_proto);
    if (request->client_helo)
	myfree(request->client_helo);
    if (request->sasl_method)
	myfree(request->sasl_method);
    if (request->sasl_username)
	myfree(request->sasl_username);
    if (request->sasl_sender)
	myfree(request->sasl_sender);
    if (request->log_ident)
	myfree(request->log_ident);
    if (request->rewrite_context)
	myfree(request->rewrite_context);
    if (request->dsn_envid)
	myfree(request->dsn_envid);
    myfree((void *) request);
}

/* deliver_request_read - create and read delivery request */

DELIVER_REQUEST *deliver_request_read(VSTREAM *stream)
{
    DELIVER_REQUEST *request;

    /*
     * Tell the queue manager that we are ready for this request.
     */
    if (deliver_request_initial(stream) != 0)
	return (0);

    /*
     * Be prepared for the queue manager to change its mind after contacting
     * us. This can happen when a transport or host goes bad.
     */
    (void) read_wait(vstream_fileno(stream), -1);
    if (peekfd(vstream_fileno(stream)) <= 0)
	return (0);

    /*
     * Allocate and read the queue manager's delivery request.
     */
#define XXX_DEFER_STATUS	-1

    request = deliver_request_alloc();
    if (deliver_request_get(stream, request) < 0) {
	deliver_request_done(stream, request, XXX_DEFER_STATUS);
	request = 0;
    }
    return (request);
}

/* deliver_request_done - finish delivery request */

int     deliver_request_done(VSTREAM *stream, DELIVER_REQUEST *request, int status)
{
    int     err;

    err = deliver_request_final(stream, request, status);
    deliver_request_free(request);
    return (err);
}
