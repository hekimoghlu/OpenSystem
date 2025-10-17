/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 1, 2025.
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
#include <errno.h>

/* Utility library. */

#include <msg.h>
#include <vstring.h>
#include <mymalloc.h>

/* Global library. */

#include <cleanup_user.h>
#include <mail_queue.h>
#include <mail_proto.h>
#include <bounce.h>
#include <mail_params.h>
#include <mail_stream.h>
#include <mail_flow.h>
#include <rec_type.h>
#include <smtputf8.h>

/* Milter library. */

#include <milter.h>

/* Application-specific. */

#include "cleanup.h"

/* cleanup_open - open queue file and initialize */

CLEANUP_STATE *cleanup_open(VSTREAM *src)
{
    CLEANUP_STATE *state;
    static const char *log_queues[] = {
	MAIL_QUEUE_DEFER,
	MAIL_QUEUE_BOUNCE,
	MAIL_QUEUE_TRACE,
	0,
    };
    const char **cpp;

    /*
     * Initialize private state.
     */
    state = cleanup_state_alloc(src);

    /*
     * Open the queue file. Save the queue file name in a global variable, so
     * that the runtime error handler can clean up in case of problems.
     * 
     * XXX For now, a lot of detail is frozen that could be more useful if it
     * were made configurable.
     */
    state->queue_name = mystrdup(MAIL_QUEUE_INCOMING);
    state->handle = mail_stream_file(state->queue_name,
				   MAIL_CLASS_PUBLIC, var_queue_service, 0);
    state->dst = state->handle->stream;
    cleanup_path = mystrdup(VSTREAM_PATH(state->dst));
    state->queue_id = mystrdup(state->handle->id);
    if (msg_verbose)
	msg_info("cleanup_open: open %s", cleanup_path);

    /*
     * If there is a time to get rid of spurious log files, this is it. The
     * down side is that this costs performance for every message, while the
     * probability of spurious log files is quite low.
     * 
     * XXX The defer logfile is deleted when the message is moved into the
     * active queue. We must also remove it now, otherwise mailq produces
     * nonsense.
     */
    for (cpp = log_queues; *cpp; cpp++) {
	if (mail_queue_remove(*cpp, state->queue_id) == 0)
	    msg_warn("%s: removed spurious %s log", *cpp, state->queue_id);
	else if (errno != ENOENT)
	    msg_fatal("%s: remove %s log: %m", *cpp, state->queue_id);
    }
    return (state);
}

/* cleanup_control - process client options */

void    cleanup_control(CLEANUP_STATE *state, int flags)
{

    /*
     * If the client requests us to do the bouncing in case of problems,
     * throw away the input only in case of real show-stopper errors, such as
     * unrecognizable data (which should never happen) or insufficient space
     * for the queue file (which will happen occasionally). Otherwise,
     * discard input after any lethal error. See the CLEANUP_OUT_OK() macro
     * definition.
     */
    if (msg_verbose)
	msg_info("cleanup flags = %s", cleanup_strflags(flags));
    if ((state->flags = flags) & CLEANUP_FLAG_BOUNCE) {
	state->err_mask = CLEANUP_STAT_MASK_INCOMPLETE;
    } else {
	state->err_mask = ~0;
    }
    if (state->flags & CLEANUP_FLAG_SMTPUTF8)
	state->smtputf8 = SMTPUTF8_FLAG_REQUESTED;
}

/* cleanup_flush - finish queue file */

int     cleanup_flush(CLEANUP_STATE *state)
{
    int     status;
    char   *junk;
    VSTRING *trace_junk;

    /*
     * Raise these errors only if we examined all queue file records.
     */
    if (CLEANUP_OUT_OK(state)) {
	if (state->recip == 0)
	    state->errs |= CLEANUP_STAT_RCPT;
	if ((state->flags & CLEANUP_FLAG_END_SEEN) == 0)
	    state->errs |= CLEANUP_STAT_BAD;
    }

    /*
     * Status sanitization. Always report success when the discard flag was
     * raised by some user-specified access rule.
     */
    if (state->flags & CLEANUP_FLAG_DISCARD)
	state->errs = 0;

    /*
     * Apply external mail filter.
     * 
     * XXX Include test for a built-in action to tempfail this message.
     */
    if (CLEANUP_MILTER_OK(state)) {
	if (state->milters)
	    cleanup_milter_inspect(state, state->milters);
	else if (cleanup_milters) {
	    cleanup_milter_emul_data(state, cleanup_milters);
	    if (CLEANUP_MILTER_OK(state))
		cleanup_milter_inspect(state, cleanup_milters);
	}
    }

    /*
     * Update the preliminary message size and count fields with the actual
     * values.
     */
    if (CLEANUP_OUT_OK(state))
	cleanup_final(state);

    /*
     * If there was an error that requires us to generate a bounce message
     * (mail submitted with the Postfix sendmail command, mail forwarded by
     * the local(8) delivery agent, or mail re-queued with "postsuper -r"),
     * send a bounce notification, reset the error flags in case of success,
     * and request deletion of the the incoming queue file and of the
     * optional DSN SUCCESS records from virtual alias expansion.
     * 
     * XXX It would make no sense to knowingly report success after we already
     * have bounced all recipients, especially because the information in the
     * DSN SUCCESS notice is completely redundant compared to the information
     * in the bounce notice (however, both may be incomplete when the queue
     * file size would exceed the safety limit).
     * 
     * An alternative is to keep the DSN SUCCESS records and to delegate bounce
     * notification to the queue manager, just like we already delegate
     * success notification. This requires that we leave the undeliverable
     * message in the incoming queue; versions up to 20050726 did exactly
     * that. Unfortunately, this broke with over-size queue files, because
     * the queue manager cannot handle incomplete queue files (and it should
     * not try to do so).
     */
#define CAN_BOUNCE() \
	((state->errs & CLEANUP_STAT_MASK_CANT_BOUNCE) == 0 \
	    && state->sender != 0 \
	    && (state->flags & CLEANUP_FLAG_BOUNCE) != 0)

    if (state->errs != 0 && CAN_BOUNCE())
	cleanup_bounce(state);

    /*
     * Optionally, place the message on hold, but only if the message was
     * received successfully and only if it's not being discarded for other
     * reasons. This involves renaming the queue file before "finishing" it
     * (or else the queue manager would grab it too early) and updating our
     * own idea of the queue file name for error recovery and for error
     * reporting purposes.
     * 
     * XXX Include test for a built-in action to tempfail this message.
     */
    if (state->errs == 0 && (state->flags & CLEANUP_FLAG_DISCARD) == 0) {
	if ((state->flags & CLEANUP_FLAG_HOLD) != 0
#ifdef DELAY_ACTION
	    || state->defer_delay > 0
#endif
	    ) {
	    myfree(state->queue_name);
#ifdef DELAY_ACTION
	    state->queue_name = mystrdup((state->flags & CLEANUP_FLAG_HOLD) ?
				     MAIL_QUEUE_HOLD : MAIL_QUEUE_DEFERRED);
#else
	    state->queue_name = mystrdup(MAIL_QUEUE_HOLD);
#endif
	    mail_stream_ctl(state->handle,
			    CA_MAIL_STREAM_CTL_QUEUE(state->queue_name),
			    CA_MAIL_STREAM_CTL_CLASS((char *) 0),
			    CA_MAIL_STREAM_CTL_SERVICE((char *) 0),
#ifdef DELAY_ACTION
			    CA_MAIL_STREAM_CTL_DELAY(state->defer_delay),
#endif
			    CA_MAIL_STREAM_CTL_END);
	    junk = cleanup_path;
	    cleanup_path = mystrdup(VSTREAM_PATH(state->handle->stream));
	    myfree(junk);

	    /*
	     * XXX: When delivering to a non-incoming queue, do not consume
	     * in_flow tokens. Unfortunately we can't move the code that
	     * consumes tokens until after the mail is received, because that
	     * would increase the risk of duplicate deliveries (RFC 1047).
	     */
	    (void) mail_flow_put(1);
	}
	state->errs = mail_stream_finish(state->handle, (VSTRING *) 0);
    } else {

	/*
	 * XXX: When discarding mail, should we consume in_flow tokens? See
	 * also the comments above for mail that is placed on hold.
	 */
#if 0
	(void) mail_flow_put(1);
#endif
	mail_stream_cleanup(state->handle);
    }
    state->handle = 0;
    state->dst = 0;

    /*
     * If there was an error, or if the message must be discarded for other
     * reasons, remove the queue file and the optional trace file with DSN
     * SUCCESS records from virtual alias expansion.
     */
    if (state->errs != 0 || (state->flags & CLEANUP_FLAG_DISCARD) != 0) {
	if (cleanup_trace_path)
	    (void) REMOVE(vstring_str(cleanup_trace_path));
	if (REMOVE(cleanup_path))
	    msg_warn("remove %s: %m", cleanup_path);
    }

    /*
     * Make sure that our queue file will not be deleted by the error handler
     * AFTER we have taken responsibility for delivery. Better to deliver
     * twice than to lose mail.
     */
    trace_junk = cleanup_trace_path;
    cleanup_trace_path = 0;			/* don't delete upon error */
    junk = cleanup_path;
    cleanup_path = 0;				/* don't delete upon error */

    if (trace_junk)
	vstring_free(trace_junk);
    myfree(junk);

    /*
     * Cleanup internal state. This is simply complementary to the
     * initializations at the beginning of cleanup_open().
     */
    if (msg_verbose)
	msg_info("cleanup_flush: status %d", state->errs);
    status = state->errs;
    return (status);
}

/* cleanup_free - pay the last respects */

void    cleanup_free(CLEANUP_STATE *state)
{

    /*
     * Emulate disconnect event. CLEANUP_FLAG_MILTER may be turned off after
     * we have started.
     */
    if (cleanup_milters != 0 && state->milters == 0)
	milter_disc_event(cleanup_milters);
    cleanup_state_free(state);
}
