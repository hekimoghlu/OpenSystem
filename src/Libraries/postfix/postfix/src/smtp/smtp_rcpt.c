/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 4, 2023.
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
/* System  library. */

#include <sys_defs.h>
#include <stdlib.h>			/* smtp_rcpt_cleanup  */
#include <string.h>

/* Utility  library. */

#include <msg.h>
#include <stringops.h>
#include <mymalloc.h>

/* Global library. */

#include <mail_params.h>
#include <deliver_request.h>		/* smtp_rcpt_done */
#include <deliver_completed.h>		/* smtp_rcpt_done */
#include <sent.h>			/* smtp_rcpt_done */
#include <dsn_mask.h>			/* smtp_rcpt_done */

/* Application-specific. */

#include <smtp.h>

/* smtp_rcpt_done - mark recipient as done or else */

void    smtp_rcpt_done(SMTP_STATE *state, SMTP_RESP *resp, RECIPIENT *rcpt)
{
    DELIVER_REQUEST *request = state->request;
    SMTP_SESSION *session = state->session;
    SMTP_ITERATOR *iter = state->iterator;
    DSN_BUF *why = state->why;
    const char *dsn_action = "relayed";
    int     status;

    /*
     * Assume this was intermediate delivery when the server announced DSN
     * support, and don't send a DSN "SUCCESS" notification.
     */
    if (session->features & SMTP_FEATURE_DSN)
	rcpt->dsn_notify &= ~DSN_NOTIFY_SUCCESS;

    /*
     * Assume this was final delivery when the LMTP server announced no DSN
     * support. In backwards compatibility mode, send a "relayed" instead of
     * a "delivered" DSN "SUCCESS" notification. Do not attempt to "simplify"
     * the expression. The redundancy is for clarity. It is trivially
     * eliminated by the compiler. There is no need to sacrifice clarity for
     * the sake of "performance".
     */
    if ((session->features & SMTP_FEATURE_DSN) == 0
	&& !smtp_mode
	&& var_lmtp_assume_final != 0)
	dsn_action = "delivered";

    /*
     * Report success and delete the recipient from the delivery request.
     * Defer if the success can't be reported.
     * 
     * Note: the DSN action is ignored in case of address probes.
     */
    dsb_update(why, resp->dsn, dsn_action, DSB_MTYPE_DNS, STR(iter->host),
	       DSB_DTYPE_SMTP, resp->str, "%s", resp->str);

    status = sent(DEL_REQ_TRACE_FLAGS(request->flags),
		  request->queue_id, &request->msg_stats, rcpt,
		  session->namaddrport, DSN_FROM_DSN_BUF(why));
    if (status == 0)
	if (request->flags & DEL_REQ_FLAG_SUCCESS)
	    deliver_completed(state->src, rcpt->offset);
    SMTP_RCPT_DROP(state, rcpt);
    state->status |= status;
}

/* smtp_rcpt_cleanup_callback - qsort callback */

static int smtp_rcpt_cleanup_callback(const void *a, const void *b)
{
    return (((RECIPIENT *) a)->u.status - ((RECIPIENT *) b)->u.status);
}

/* smtp_rcpt_cleanup - purge completed recipients from request */

void    smtp_rcpt_cleanup(SMTP_STATE *state)
{
    RECIPIENT_LIST *rcpt_list = &state->request->rcpt_list;
    RECIPIENT *rcpt;

    /*
     * Sanity checks.
     */
    if (state->rcpt_drop + state->rcpt_keep != state->rcpt_left)
	msg_panic("smtp_rcpt_cleanup: recipient count mismatch: %d+%d!=%d",
		  state->rcpt_drop, state->rcpt_keep, state->rcpt_left);

    /*
     * Recipients marked KEEP sort before recipients marked DROP. Skip the
     * sorting in the common case that all recipients are marked the same.
     */
    if (state->rcpt_drop > 0 && state->rcpt_keep > 0)
	qsort((void *) rcpt_list->info, state->rcpt_left,
	      sizeof(rcpt_list->info[0]), smtp_rcpt_cleanup_callback);

    /*
     * Truncate the recipient list and unmark the left-over recipients.
     */
    state->rcpt_left = state->rcpt_keep;
    for (rcpt = rcpt_list->info; rcpt < rcpt_list->info + state->rcpt_left; rcpt++)
	rcpt->u.status = 0;
    state->rcpt_drop = state->rcpt_keep = 0;
}
