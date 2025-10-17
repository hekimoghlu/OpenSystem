/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 13, 2022.
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
#include <unistd.h>

/* Utility library. */

#include <msg.h>
#include <vstream.h>
#include <set_eugid.h>

/* Global library. */

#include <mail_queue.h>
#include <mail_params.h>
#include <deliver_request.h>
#include <mark_corrupt.h>

/* mark_corrupt - mark queue file as corrupt */

int     mark_corrupt(VSTREAM *src)
{
    const char *myname = "mark_corrupt";
    uid_t   saved_uid;
    gid_t   saved_gid;

    /*
     * If not running as the mail system, change privileges first.
     */
    if ((saved_uid = geteuid()) != var_owner_uid) {
	saved_gid = getegid();
	set_eugid(var_owner_uid, var_owner_gid);
    }

    /*
     * For now, the result value is -1; this may become a bit mask, or
     * something even more advanced than that, when the delivery status
     * becomes more than just done/deferred.
     */
    msg_warn("corrupted queue file: %s", VSTREAM_PATH(src));
    if (fchmod(vstream_fileno(src), MAIL_QUEUE_STAT_CORRUPT))
	msg_fatal("%s: fchmod %s: %m", myname, VSTREAM_PATH(src));

    /*
     * Restore privileges.
     */
    if (saved_uid != var_owner_uid)
	set_eugid(saved_uid, saved_gid);

    return (DEL_STAT_DEFER);
}
