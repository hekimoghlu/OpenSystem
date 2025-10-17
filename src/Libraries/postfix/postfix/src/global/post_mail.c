/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 25, 2022.
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
#include <sys/time.h>
#include <stdlib.h>			/* 44BSD stdarg.h uses abort() */
#include <stdarg.h>
#include <string.h>

/* Utility library. */

#include <msg.h>
#include <vstream.h>
#include <vstring.h>
#include <mymalloc.h>
#include <events.h>

/* Global library. */

#include <mail_params.h>
#include <record.h>
#include <rec_type.h>
#include <mail_proto.h>
#include <cleanup_user.h>
#include <post_mail.h>
#include <mail_date.h>

 /*
  * Call-back state for asynchronous connection requests.
  */
typedef struct {
    char   *sender;
    char   *recipient;
    int     source_class;
    int     trace_flags;
    int     utf8_flags;
    POST_MAIL_NOTIFY notify;
    void   *context;
    VSTREAM *stream;
    VSTRING *queue_id;
} POST_MAIL_STATE;

 /*
  * Call-back state for asynchronous close requests.
  */
typedef struct {
    int     status;
    VSTREAM *stream;
    POST_MAIL_FCLOSE_NOTIFY notify;
    void   *context;
} POST_MAIL_FCLOSE_STATE;

/* post_mail_init - initial negotiations */

static void post_mail_init(VSTREAM *stream, const char *sender,
			           const char *recipient,
			           int source_class, int trace_flags,
			           int utf8_flags, VSTRING *queue_id)
{
    VSTRING *id = queue_id ? queue_id : vstring_alloc(100);
    struct timeval now;
    const char *date;
    int     cleanup_flags =
    int_filt_flags(source_class) | CLEANUP_FLAG_MASK_INTERNAL
    | smtputf8_autodetect(source_class)
    | ((utf8_flags & SMTPUTF8_FLAG_REQUESTED) ? CLEANUP_FLAG_SMTPUTF8 : 0);

    GETTIMEOFDAY(&now);
    date = mail_date(now.tv_sec);

    /*
     * XXX Don't flush buffers while sending the initial message records.
     * That would cause deadlock between verify(8) and cleanup(8) servers.
     */
    vstream_control(stream, VSTREAM_CTL_BUFSIZE, 2 * VSTREAM_BUFSIZE,
		    VSTREAM_CTL_END);

    /*
     * Negotiate with the cleanup service. Give up if we can't agree.
     */
    if (attr_scan(stream, ATTR_FLAG_STRICT,
		  RECV_ATTR_STR(MAIL_ATTR_QUEUEID, id),
		  ATTR_TYPE_END) != 1
	|| attr_print(stream, ATTR_FLAG_NONE,
		      SEND_ATTR_INT(MAIL_ATTR_FLAGS, cleanup_flags),
		      ATTR_TYPE_END) != 0)
	msg_fatal("unable to contact the %s service", var_cleanup_service);

    /*
     * Generate a minimal envelope section. The cleanup service will add a
     * size record.
     */
    rec_fprintf(stream, REC_TYPE_TIME, REC_TYPE_TIME_FORMAT,
		REC_TYPE_TIME_ARG(now));
    rec_fprintf(stream, REC_TYPE_ATTR, "%s=%s",
		MAIL_ATTR_LOG_ORIGIN, MAIL_ATTR_ORG_LOCAL);
    rec_fprintf(stream, REC_TYPE_ATTR, "%s=%d",
		MAIL_ATTR_TRACE_FLAGS, trace_flags);
    rec_fputs(stream, REC_TYPE_FROM, sender);
    rec_fputs(stream, REC_TYPE_RCPT, recipient);
    rec_fputs(stream, REC_TYPE_MESG, "");

    /*
     * Do the Received: and Date: header lines. This allows us to shave a few
     * cycles by using the expensive date conversion result for both.
     */
    post_mail_fprintf(stream, "Received: by %s (%s)",
		      var_myhostname, var_mail_name);
    post_mail_fprintf(stream, "\tid %s; %s", vstring_str(id), date);
    post_mail_fprintf(stream, "Date: %s", date);
    if (queue_id == 0)
	vstring_free(id);
}

/* post_mail_fopen - prepare for posting a message */

VSTREAM *post_mail_fopen(const char *sender, const char *recipient,
			         int source_class, int trace_flags,
			         int utf8_flags, VSTRING *queue_id)
{
    VSTREAM *stream;

    stream = mail_connect_wait(MAIL_CLASS_PUBLIC, var_cleanup_service);
    post_mail_init(stream, sender, recipient, source_class, trace_flags,
		   utf8_flags, queue_id);
    return (stream);
}

/* post_mail_fopen_nowait - prepare for posting a message */

VSTREAM *post_mail_fopen_nowait(const char *sender, const char *recipient,
				        int source_class, int trace_flags,
				        int utf8_flags, VSTRING *queue_id)
{
    VSTREAM *stream;

    if ((stream = mail_connect(MAIL_CLASS_PUBLIC, var_cleanup_service,
			       BLOCKING)) != 0)
	post_mail_init(stream, sender, recipient, source_class, trace_flags,
		       utf8_flags, queue_id);
    else
	msg_warn("connect to %s/%s: %m",
		 MAIL_CLASS_PUBLIC, var_cleanup_service);
    return (stream);
}

/* post_mail_open_event - handle asynchronous connection events */

static void post_mail_open_event(int event, void *context)
{
    POST_MAIL_STATE *state = (POST_MAIL_STATE *) context;
    const char *myname = "post_mail_open_event";

    switch (event) {

	/*
	 * Initial server reply. Stop the watchdog timer, disable further
	 * read events that end up calling this function, and notify the
	 * requestor.
	 */
    case EVENT_READ:
	if (msg_verbose)
	    msg_info("%s: read event", myname);
	event_cancel_timer(post_mail_open_event, context);
	event_disable_readwrite(vstream_fileno(state->stream));
	non_blocking(vstream_fileno(state->stream), BLOCKING);
	post_mail_init(state->stream, state->sender,
		       state->recipient, state->source_class,
		       state->trace_flags, state->utf8_flags,
		       state->queue_id);
	myfree(state->sender);
	myfree(state->recipient);
	state->notify(state->stream, state->context);
	myfree((void *) state);
	return;

	/*
	 * No connection or no initial reply within a conservative time
	 * limit. The system is broken and we give up.
	 */
    case EVENT_TIME:
	if (state->stream) {
	    msg_warn("timeout connecting to service: %s", var_cleanup_service);
	    event_disable_readwrite(vstream_fileno(state->stream));
	    vstream_fclose(state->stream);
	} else {
	    msg_warn("connect to service: %s: %m", var_cleanup_service);
	}
	myfree(state->sender);
	myfree(state->recipient);
	state->notify((VSTREAM *) 0, state->context);
	myfree((void *) state);
	return;

	/*
	 * Some exception.
	 */
    case EVENT_XCPT:
	msg_warn("error connecting to service: %s", var_cleanup_service);
	event_cancel_timer(post_mail_open_event, context);
	event_disable_readwrite(vstream_fileno(state->stream));
	vstream_fclose(state->stream);
	myfree(state->sender);
	myfree(state->recipient);
	state->notify((VSTREAM *) 0, state->context);
	myfree((void *) state);
	return;

	/*
	 * Broken software or hardware.
	 */
    default:
	msg_panic("%s: unknown event type %d", myname, event);
    }
}

/* post_mail_fopen_async - prepare for posting a message */

void    post_mail_fopen_async(const char *sender, const char *recipient,
			              int source_class, int trace_flags,
			              int utf8_flags, VSTRING *queue_id,
			              void (*notify) (VSTREAM *, void *),
			              void *context)
{
    VSTREAM *stream;
    POST_MAIL_STATE *state;

    stream = mail_connect(MAIL_CLASS_PUBLIC, var_cleanup_service, NON_BLOCKING);
    state = (POST_MAIL_STATE *) mymalloc(sizeof(*state));
    state->sender = mystrdup(sender);
    state->recipient = mystrdup(recipient);
    state->source_class = source_class;
    state->trace_flags = trace_flags;
    state->utf8_flags = utf8_flags;
    state->notify = notify;
    state->context = context;
    state->stream = stream;
    state->queue_id = queue_id;

    /*
     * To keep interfaces as simple as possible we report all errors via the
     * same interface as all successes.
     */
    if (stream != 0) {
	event_enable_read(vstream_fileno(stream), post_mail_open_event,
			  (void *) state);
	event_request_timer(post_mail_open_event, (void *) state,
			    var_daemon_timeout);
    } else {
	event_request_timer(post_mail_open_event, (void *) state, 0);
    }
}

/* post_mail_fprintf - format and send message content */

int     post_mail_fprintf(VSTREAM *cleanup, const char *format,...)
{
    int     status;
    va_list ap;

    va_start(ap, format);
    status = rec_vfprintf(cleanup, REC_TYPE_NORM, format, ap);
    va_end(ap);
    return (status != REC_TYPE_NORM ? CLEANUP_STAT_WRITE : 0);
}

/* post_mail_buffer - send pre-formatted buffer */

int     post_mail_buffer(VSTREAM *cleanup, const char *buf, int len)
{
    return (rec_put(cleanup, REC_TYPE_NORM, buf, len) != REC_TYPE_NORM ?
	    CLEANUP_STAT_WRITE : 0);
}

/* post_mail_fputs - send pre-formatted message content */

int     post_mail_fputs(VSTREAM *cleanup, const char *str)
{
    ssize_t len = str ? strlen(str) : 0;

    return (rec_put(cleanup, REC_TYPE_NORM, str, len) != REC_TYPE_NORM ?
	    CLEANUP_STAT_WRITE : 0);
}

/* post_mail_fclose - finish posting of message */

int     post_mail_fclose(VSTREAM *cleanup)
{
    int     status = 0;

    /*
     * Send the message end marker only when there were no errors.
     */
    if (vstream_ferror(cleanup) != 0) {
	status = CLEANUP_STAT_WRITE;
    } else {
	rec_fputs(cleanup, REC_TYPE_XTRA, "");
	rec_fputs(cleanup, REC_TYPE_END, "");
	if (vstream_fflush(cleanup)
	    || attr_scan(cleanup, ATTR_FLAG_MISSING,
			 RECV_ATTR_INT(MAIL_ATTR_STATUS, &status),
			 ATTR_TYPE_END) != 1)
	    status = CLEANUP_STAT_WRITE;
    }
    (void) vstream_fclose(cleanup);
    return (status);
}

/* post_mail_fclose_event - event handler */

static void post_mail_fclose_event(int event, void *context)
{
    POST_MAIL_FCLOSE_STATE *state = (POST_MAIL_FCLOSE_STATE *) context;
    int     status = state->status;

    switch (event) {

	/*
	 * Final server reply. Pick up the completion status.
	 */
    case EVENT_READ:
	if (status == 0) {
	    if (vstream_ferror(state->stream) != 0
		|| attr_scan(state->stream, ATTR_FLAG_MISSING,
			     ATTR_TYPE_INT, MAIL_ATTR_STATUS, &status,
			     ATTR_TYPE_END) != 1)
		status = CLEANUP_STAT_WRITE;
	}
	break;

	/*
	 * No response or error.
	 */
    default:
	msg_warn("error talking to service: %s", var_cleanup_service);
	status = CLEANUP_STAT_WRITE;
	break;
    }

    /*
     * Stop the watchdog timer, and disable further read events that end up
     * calling this function.
     */
    event_cancel_timer(post_mail_fclose_event, context);
    event_disable_readwrite(vstream_fileno(state->stream));

    /*
     * Notify the requestor and clean up.
     */
    state->notify(status, state->context);
    (void) vstream_fclose(state->stream);
    myfree((void *) state);
}

/* post_mail_fclose_async - finish posting of message */

void    post_mail_fclose_async(VSTREAM *stream,
			         void (*notify) (int status, void *context),
			               void *context)
{
    POST_MAIL_FCLOSE_STATE *state;
    int     status = 0;


    /*
     * Send the message end marker only when there were no errors.
     */
    if (vstream_ferror(stream) != 0) {
	status = CLEANUP_STAT_WRITE;
    } else {
	rec_fputs(stream, REC_TYPE_XTRA, "");
	rec_fputs(stream, REC_TYPE_END, "");
	if (vstream_fflush(stream))
	    status = CLEANUP_STAT_WRITE;
    }

    /*
     * Bundle up the suspended state.
     */
    state = (POST_MAIL_FCLOSE_STATE *) mymalloc(sizeof(*state));
    state->status = status;
    state->stream = stream;
    state->notify = notify;
    state->context = context;

    /*
     * To keep interfaces as simple as possible we report all errors via the
     * same interface as all successes.
     */
    if (status == 0) {
	event_enable_read(vstream_fileno(stream), post_mail_fclose_event,
			  (void *) state);
	event_request_timer(post_mail_fclose_event, (void *) state,
			    var_daemon_timeout);
    } else {
	event_request_timer(post_mail_fclose_event, (void *) state, 0);
    }
}
