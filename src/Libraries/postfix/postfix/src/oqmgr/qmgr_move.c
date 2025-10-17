/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 23, 2024.
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
#include <utime.h>
#include <errno.h>

/* Utility library. */

#include <msg.h>
#include <scan_dir.h>
#include <recipient_list.h>

/* Global library. */

#include <mail_queue.h>
#include <mail_scan_dir.h>

/* Application-specific. */

#include "qmgr.h"

/* qmgr_move - move queue entries to another queue, leave bad files alone */

void    qmgr_move(const char *src_queue, const char *dst_queue,
		          time_t time_stamp)
{
    const char *myname = "qmgr_move";
    SCAN_DIR *queue_dir;
    char   *queue_id;
    struct utimbuf tbuf;
    const char *path;

    if (strcmp(src_queue, dst_queue) == 0)
	msg_panic("%s: source queue %s is destination", myname, src_queue);
    if (msg_verbose)
	msg_info("start move queue %s -> %s", src_queue, dst_queue);

    queue_dir = scan_dir_open(src_queue);
    while ((queue_id = mail_scan_dir_next(queue_dir)) != 0) {
	if (mail_queue_id_ok(queue_id)) {
	    if (time_stamp > 0) {
		tbuf.actime = tbuf.modtime = time_stamp;
		path = mail_queue_path((VSTRING *) 0, src_queue, queue_id);
		if (utime(path, &tbuf) < 0) {
		    if (errno != ENOENT)
			msg_fatal("%s: update %s time stamps: %m", myname, path);
		    msg_warn("%s: update %s time stamps: %m", myname, path);
		    continue;
		}
	    }
	    if (mail_queue_rename(queue_id, src_queue, dst_queue)) {
		if (errno != ENOENT)
		    msg_fatal("%s: rename %s from %s to %s: %m",
			      myname, queue_id, src_queue, dst_queue);
		msg_warn("%s: rename %s from %s to %s: %m",
			 myname, queue_id, src_queue, dst_queue);
		continue;
	    }
	    if (msg_verbose)
		msg_info("%s: moved %s from %s to %s",
			 myname, queue_id, src_queue, dst_queue);
	} else {
	    msg_warn("%s: ignored: queue %s id %s",
		     myname, src_queue, queue_id);
	}
    }
    scan_dir_close(queue_dir);

    if (msg_verbose)
	msg_info("end move queue %s -> %s", src_queue, dst_queue);
}
