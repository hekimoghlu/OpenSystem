/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 28, 2022.
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
#include <stdio.h>
#include <string.h>

/* Utility library. */

#include <msg.h>
#include <vstring.h>

/* Global library. */

#include <mail_params.h>
#include <mail_proto.h>
#include <log_adhoc.h>
#include <rcpt_print.h>
#include <dsn_print.h>
#include <trace.h>

/* trace_append - append to message delivery record */

int     trace_append(int flags, const char *id, MSG_STATS *stats,
		             RECIPIENT *rcpt, const char *relay,
		             DSN *dsn)
{
    VSTRING *why = vstring_alloc(100);
    DSN     my_dsn = *dsn;
    int     req_stat;

    /*
     * User-requested address verification, verbose delivery, or DSN SUCCESS
     * notification.
     */
    if (strcmp(relay, NO_RELAY_AGENT) != 0)
	vstring_sprintf(why, "delivery via %s: ", relay);
    vstring_strcat(why, my_dsn.reason);
    my_dsn.reason = vstring_str(why);

    if (mail_command_client(MAIL_CLASS_PRIVATE, var_trace_service,
			    SEND_ATTR_INT(MAIL_ATTR_NREQ, BOUNCE_CMD_APPEND),
			    SEND_ATTR_INT(MAIL_ATTR_FLAGS, flags),
			    SEND_ATTR_STR(MAIL_ATTR_QUEUEID, id),
			    SEND_ATTR_FUNC(rcpt_print, (void *) rcpt),
			    SEND_ATTR_FUNC(dsn_print, (void *) &my_dsn),
			    ATTR_TYPE_END) != 0) {
	msg_warn("%s: %s service failure", id, var_trace_service);
	req_stat = -1;
    } else {
	if (flags & DEL_REQ_FLAG_USR_VRFY)
	    log_adhoc(id, stats, rcpt, relay, dsn, my_dsn.action);
	req_stat = 0;
    }
    vstring_free(why);
    return (req_stat);
}

/* trace_flush - deliver delivery record to the sender */

int     trace_flush(int flags, const char *queue, const char *id,
		            const char *encoding, const char *sender,
		            const char *dsn_envid, int dsn_ret)
{
    if (mail_command_client(MAIL_CLASS_PRIVATE, var_trace_service,
			    SEND_ATTR_INT(MAIL_ATTR_NREQ, BOUNCE_CMD_TRACE),
			    SEND_ATTR_INT(MAIL_ATTR_FLAGS, flags),
			    SEND_ATTR_STR(MAIL_ATTR_QUEUE, queue),
			    SEND_ATTR_STR(MAIL_ATTR_QUEUEID, id),
			    SEND_ATTR_STR(MAIL_ATTR_ENCODING, encoding),
			    SEND_ATTR_STR(MAIL_ATTR_SENDER, sender),
			    SEND_ATTR_STR(MAIL_ATTR_DSN_ENVID, dsn_envid),
			    SEND_ATTR_INT(MAIL_ATTR_DSN_RET, dsn_ret),
			    ATTR_TYPE_END) == 0) {
	return (0);
    } else {
	return (-1);
    }
}
