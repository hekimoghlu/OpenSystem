/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 19, 2022.
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
#include <string.h>

/* Utility library. */

#include <msg.h>
#include <vstring.h>
#include <stringops.h>

/* Global library. */

#include <mail_params.h>
#include <mail_proto.h>
#include <verify_clnt.h>
#include <log_adhoc.h>
#include <verify.h>

/* verify_append - update address verification database */

int     verify_append(const char *queue_id, MSG_STATS *stats,
		              RECIPIENT *recipient, const char *relay,
		              DSN *dsn, int vrfy_stat)
{
    int     req_stat;
    DSN     my_dsn = *dsn;

    /*
     * Impedance adaptor between bounce/defer/sent and verify_clnt.
     * 
     * XXX No DSN check; this routine is called from bounce/defer/sent, which
     * know what the DSN initial digit should look like.
     * 
     * XXX vrfy_stat is competely redundant because of dsn.
     */
    if (var_verify_neg_cache || vrfy_stat == DEL_RCPT_STAT_OK) {
	req_stat = verify_clnt_update(recipient->orig_addr, vrfy_stat,
				      my_dsn.reason);
	/* Two verify updates for one verify request! */
	if (req_stat == VRFY_STAT_OK
	  && strcasecmp_utf8(recipient->address, recipient->orig_addr) != 0)
	    req_stat = verify_clnt_update(recipient->address, vrfy_stat,
					  my_dsn.reason);
    } else {
	my_dsn.action = "undeliverable-but-not-cached";
	req_stat = VRFY_STAT_OK;
    }
    if (req_stat == VRFY_STAT_OK) {
	log_adhoc(queue_id, stats, recipient, relay, dsn, my_dsn.action);
	req_stat = 0;
    } else {
	msg_warn("%s: %s service failure", queue_id, var_verify_service);
	req_stat = -1;
    }
    return (req_stat);
}
