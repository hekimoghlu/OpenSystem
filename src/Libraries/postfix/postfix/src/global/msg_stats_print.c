/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 28, 2022.
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

/* Global library. */

#include <mail_proto.h>
#include <msg_stats.h>

/* msg_stats_print - write MSG_STATS to stream */

int     msg_stats_print(ATTR_PRINT_MASTER_FN print_fn, VSTREAM *fp,
			        int flags, void *ptr)
{
    int     ret;

    /*
     * Send the entire structure. This is not only simpler but also likely to
     * be quicker than having the sender figure out what fields need to be
     * sent, converting numbers to string and back, and having the receiver
     * initialize the unused fields by hand.
     */
    ret = print_fn(fp, flags | ATTR_FLAG_MORE,
		   SEND_ATTR_DATA(MAIL_ATTR_TIME, sizeof(MSG_STATS), ptr),
		   ATTR_TYPE_END);
    return (ret);
}
