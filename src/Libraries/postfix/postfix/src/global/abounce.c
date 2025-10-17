/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 23, 2024.
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

/* Utility library. */

#include <msg.h>
#include <mymalloc.h>
#include <events.h>
#include <vstream.h>

/* Global library. */

#include <mail_params.h>
#include <mail_proto.h>
#include <abounce.h>

/* Application-specific. */

 /*
  * Each bounce/defer flush/warn request is implemented by sending the
  * request to the bounce/defer server, and by creating a pseudo thread that
  * suspends itself until the server replies (or dies). Upon wakeup, the
  * pseudo thread delivers the request completion status to the application
  * and destroys itself. The structure below maintains all the necessary
  * request state while the pseudo thread is suspended.
  */
typedef struct {
    int     command;			/* bounce request type */
    int     flags;			/* bounce options */
    char   *id;				/* queue ID for logging */
    ABOUNCE_FN callback;		/* application callback */
    void   *context;			/* application context */
    VSTREAM *fp;			/* server I/O handle */
} ABOUNCE;

 /*
  * Encapsulate common code.
  */
#define ABOUNCE_EVENT_ENABLE(fd, callback, context, timeout) do { \
	event_enable_read((fd), (callback), (context)); \
	event_request_timer((callback), (context), (timeout)); \
    } while (0)

#define ABOUNCE_EVENT_DISABLE(fd, callback, context) do { \
	event_cancel_timer((callback), (context)); \
	event_disable_readwrite(fd); \
    } while (0)

 /*
  * If we set the reply timeout too short, then we make the problem worse by
  * increasing overload. With 1000s timeout mail will keep flowing, but there
  * will be a large number of blocked bounce processes, and some resource is
  * likely to run out.
  */
#define ABOUNCE_TIMEOUT	1000

/* abounce_done - deliver status to application and clean up pseudo thread */

static void abounce_done(ABOUNCE *ap, int status)
{
    (void) vstream_fclose(ap->fp);
    if (status != 0 && (ap->flags & BOUNCE_FLAG_CLEAN) == 0)
	msg_info("%s: status=deferred (%s failed)", ap->id,
		 ap->command == BOUNCE_CMD_FLUSH ? "bounce" :
		 ap->command == BOUNCE_CMD_WARN ? "delay warning" :
		 ap->command == BOUNCE_CMD_VERP ? "verp" :
		 ap->command == BOUNCE_CMD_TRACE ? "trace" :
		 "whatever");
    ap->callback(status, ap->context);
    myfree(ap->id);
    myfree((void *) ap);
}

/* abounce_event - resume pseudo thread after server reply event */

static void abounce_event(int event, void *context)
{
    ABOUNCE *ap = (ABOUNCE *) context;
    int     status;

    ABOUNCE_EVENT_DISABLE(vstream_fileno(ap->fp), abounce_event, context);
    abounce_done(ap, (event != EVENT_TIME
		      && attr_scan(ap->fp, ATTR_FLAG_STRICT,
				   RECV_ATTR_INT(MAIL_ATTR_STATUS, &status),
				   ATTR_TYPE_END) == 1) ? status : -1);
}

/* abounce_request_verp - suspend pseudo thread until server reply event */

static void abounce_request_verp(const char *class, const char *service,
				         int command, int flags,
				         const char *queue, const char *id,
				         const char *encoding,
				         int smtputf8,
				         const char *sender,
				         const char *dsn_envid,
				         int dsn_ret,
				         const char *verp,
				         ABOUNCE_FN callback,
				         void *context)
{
    ABOUNCE *ap;

    /*
     * Save pseudo thread state. Connect to the server. Send the request and
     * suspend the pseudo thread until the server replies (or dies).
     */
    ap = (ABOUNCE *) mymalloc(sizeof(*ap));
    ap->command = command;
    ap->flags = flags;
    ap->id = mystrdup(id);
    ap->callback = callback;
    ap->context = context;
    ap->fp = mail_connect_wait(class, service);

    if (attr_print(ap->fp, ATTR_FLAG_NONE,
		   SEND_ATTR_INT(MAIL_ATTR_NREQ, command),
		   SEND_ATTR_INT(MAIL_ATTR_FLAGS, flags),
		   SEND_ATTR_STR(MAIL_ATTR_QUEUE, queue),
		   SEND_ATTR_STR(MAIL_ATTR_QUEUEID, id),
		   SEND_ATTR_STR(MAIL_ATTR_ENCODING, encoding),
		   SEND_ATTR_INT(MAIL_ATTR_SMTPUTF8, smtputf8),
		   SEND_ATTR_STR(MAIL_ATTR_SENDER, sender),
		   SEND_ATTR_STR(MAIL_ATTR_DSN_ENVID, dsn_envid),
		   SEND_ATTR_INT(MAIL_ATTR_DSN_RET, dsn_ret),
		   SEND_ATTR_STR(MAIL_ATTR_VERPDL, verp),
		   ATTR_TYPE_END) == 0
	&& vstream_fflush(ap->fp) == 0) {
	ABOUNCE_EVENT_ENABLE(vstream_fileno(ap->fp), abounce_event,
			     (void *) ap, ABOUNCE_TIMEOUT);
    } else {
	abounce_done(ap, -1);
    }
}

/* abounce_flush_verp - asynchronous bounce flush */

void    abounce_flush_verp(int flags, const char *queue, const char *id,
			           const char *encoding, int smtputf8,
			           const char *sender, const char *dsn_envid,
			           int dsn_ret, const char *verp,
			           ABOUNCE_FN callback,
			           void *context)
{
    abounce_request_verp(MAIL_CLASS_PRIVATE, var_bounce_service,
		      BOUNCE_CMD_VERP, flags, queue, id, encoding, smtputf8,
		       sender, dsn_envid, dsn_ret, verp, callback, context);
}

/* adefer_flush_verp - asynchronous defer flush */

void    adefer_flush_verp(int flags, const char *queue, const char *id,
			          const char *encoding, int smtputf8,
			          const char *sender, const char *dsn_envid,
			          int dsn_ret, const char *verp,
			          ABOUNCE_FN callback, void *context)
{
    flags |= BOUNCE_FLAG_DELRCPT;
    abounce_request_verp(MAIL_CLASS_PRIVATE, var_defer_service,
		      BOUNCE_CMD_VERP, flags, queue, id, encoding, smtputf8,
		       sender, dsn_envid, dsn_ret, verp, callback, context);
}

/* abounce_request - suspend pseudo thread until server reply event */

static void abounce_request(const char *class, const char *service,
			            int command, int flags,
			            const char *queue, const char *id,
			            const char *encoding, int smtputf8,
			            const char *sender,
			            const char *dsn_envid, int dsn_ret,
			            ABOUNCE_FN callback, void *context)
{
    ABOUNCE *ap;

    /*
     * Save pseudo thread state. Connect to the server. Send the request and
     * suspend the pseudo thread until the server replies (or dies).
     */
    ap = (ABOUNCE *) mymalloc(sizeof(*ap));
    ap->command = command;
    ap->flags = flags;
    ap->id = mystrdup(id);
    ap->callback = callback;
    ap->context = context;
    ap->fp = mail_connect_wait(class, service);

    if (attr_print(ap->fp, ATTR_FLAG_NONE,
		   SEND_ATTR_INT(MAIL_ATTR_NREQ, command),
		   SEND_ATTR_INT(MAIL_ATTR_FLAGS, flags),
		   SEND_ATTR_STR(MAIL_ATTR_QUEUE, queue),
		   SEND_ATTR_STR(MAIL_ATTR_QUEUEID, id),
		   SEND_ATTR_STR(MAIL_ATTR_ENCODING, encoding),
		   SEND_ATTR_INT(MAIL_ATTR_SMTPUTF8, smtputf8),
		   SEND_ATTR_STR(MAIL_ATTR_SENDER, sender),
		   SEND_ATTR_STR(MAIL_ATTR_DSN_ENVID, dsn_envid),
		   SEND_ATTR_INT(MAIL_ATTR_DSN_RET, dsn_ret),
		   ATTR_TYPE_END) == 0
	&& vstream_fflush(ap->fp) == 0) {
	ABOUNCE_EVENT_ENABLE(vstream_fileno(ap->fp), abounce_event,
			     (void *) ap, ABOUNCE_TIMEOUT);
    } else {
	abounce_done(ap, -1);
    }
}

/* abounce_flush - asynchronous bounce flush */

void    abounce_flush(int flags, const char *queue, const char *id,
		              const char *encoding, int smtputf8,
		              const char *sender, const char *dsn_envid,
		              int dsn_ret, ABOUNCE_FN callback,
		              void *context)
{
    abounce_request(MAIL_CLASS_PRIVATE, var_bounce_service, BOUNCE_CMD_FLUSH,
		    flags, queue, id, encoding, smtputf8, sender, dsn_envid,
		    dsn_ret, callback, context);
}

/* adefer_flush - asynchronous defer flush */

void    adefer_flush(int flags, const char *queue, const char *id,
		             const char *encoding, int smtputf8,
		             const char *sender, const char *dsn_envid,
		             int dsn_ret, ABOUNCE_FN callback, void *context)
{
    flags |= BOUNCE_FLAG_DELRCPT;
    abounce_request(MAIL_CLASS_PRIVATE, var_defer_service, BOUNCE_CMD_FLUSH,
		    flags, queue, id, encoding, smtputf8, sender, dsn_envid,
		    dsn_ret, callback, context);
}

/* adefer_warn - send copy of defer log to sender as warning bounce */

void    adefer_warn(int flags, const char *queue, const char *id,
		            const char *encoding, int smtputf8,
		            const char *sender, const char *dsn_envid,
		            int dsn_ret, ABOUNCE_FN callback, void *context)
{
    abounce_request(MAIL_CLASS_PRIVATE, var_defer_service, BOUNCE_CMD_WARN,
		    flags, queue, id, encoding, smtputf8, sender, dsn_envid,
		    dsn_ret, callback, context);
}

/* atrace_flush - asynchronous trace flush */

void    atrace_flush(int flags, const char *queue, const char *id,
		             const char *encoding, int smtputf8,
		             const char *sender, const char *dsn_envid,
		             int dsn_ret, ABOUNCE_FN callback, void *context)
{
    abounce_request(MAIL_CLASS_PRIVATE, var_trace_service, BOUNCE_CMD_TRACE,
		    flags, queue, id, encoding, smtputf8, sender, dsn_envid,
		    dsn_ret, callback, context);
}
