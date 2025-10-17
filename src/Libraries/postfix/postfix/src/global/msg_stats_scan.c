/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 19, 2021.
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

#include <attr.h>
#include <vstring.h>
#include <msg.h>

/* Global library. */

#include <mail_proto.h>
#include <msg_stats.h>

 /*
  * SLMs.
  */
#define STR(x) vstring_str(x)
#define LEN(x) VSTRING_LEN(x)

/* msg_stats_scan - read MSG_STATS from stream */

int     msg_stats_scan(ATTR_SCAN_MASTER_FN scan_fn, VSTREAM *fp,
		               int flags, void *ptr)
{
    MSG_STATS *stats = (MSG_STATS *) ptr;
    VSTRING *buf = vstring_alloc(sizeof(MSG_STATS) * 2);
    int     ret;

    /*
     * Receive the entire structure. This is not only simpler but also likely
     * to be quicker than having the sender figure out what fields need to be
     * sent, converting those numbers to string and back, and having the
     * receiver initialize the unused fields by hand.
     * 
     * XXX Would be nice if VSTRINGs could import a fixed-size buffer and
     * gracefully reject attempts to extend it.
     */
    ret = scan_fn(fp, flags | ATTR_FLAG_MORE,
		  RECV_ATTR_DATA(MAIL_ATTR_TIME, buf),
		  ATTR_TYPE_END);
    if (ret == 1) {
	if (LEN(buf) == sizeof(*stats)) {
	    memcpy((void *) stats, STR(buf), sizeof(*stats));
	} else {
	    msg_warn("msg_stats_scan: size mis-match: %u != %u",
		     (unsigned) LEN(buf), (unsigned) sizeof(*stats));
	    ret = (-1);
	}
    }
    vstring_free(buf);
    return (ret);
}
