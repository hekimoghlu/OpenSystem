/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 12, 2025.
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
/* System libraries. */

#include <sys_defs.h>
#include <sys/stat.h>
#include <time.h>
#include <errno.h>

/* Utility library. */

#include <msg.h>
#include <warn_stat.h>

/* Global library. */

#include "mail_queue.h"
#include "mail_open_ok.h"

/* mail_open_ok - see if this file is OK to open */

int     mail_open_ok(const char *queue_name, const char *queue_id,
		             struct stat * statp, const char **path)
{
    if (mail_queue_name_ok(queue_name) == 0) {
	msg_warn("bad mail queue name: %s", queue_name);
	return (MAIL_OPEN_NO);
    }
    if (mail_queue_id_ok(queue_id) == 0)
	return (MAIL_OPEN_NO);


    /*
     * I really would like to look up the file attributes *after* opening the
     * file so that we could save one directory traversal on systems without
     * name-to-inode cache. However, we don't necessarily always want to open
     * the file.
     */
    *path = mail_queue_path((VSTRING *) 0, queue_name, queue_id);

    if (lstat(*path, statp) < 0) {
	if (errno != ENOENT)
	    msg_warn("%s: %m", *path);
	return (MAIL_OPEN_NO);
    }
    if (!S_ISREG(statp->st_mode)) {
	msg_warn("%s: uid %ld: not a regular file", *path, (long) statp->st_uid);
	return (MAIL_OPEN_NO);
    }
    if ((statp->st_mode & S_IRWXU) != MAIL_QUEUE_STAT_READY)
	return (MAIL_OPEN_NO);

    /*
     * Workaround for spurious "file has 2 links" warnings in showq. As
     * kernels have evolved from non-interruptible system calls towards
     * fine-grained locks, the showq command has become likely to observe a
     * file while the queue manager is in the middle of renaming it, at a
     * time when the file has links to both the old and new name. We now log
     * the warning only when the condition appears to be persistent.
     */
#define MINUTE_SECONDS	60			/* XXX should be centralized */

    if (statp->st_nlink > 1) {
	if (msg_verbose)
	    msg_info("%s: uid %ld: file has %d links", *path,
		     (long) statp->st_uid, (int) statp->st_nlink);
	else if (statp->st_ctime < time((time_t *) 0) - MINUTE_SECONDS)
	    msg_warn("%s: uid %ld: file has %d links", *path,
		     (long) statp->st_uid, (int) statp->st_nlink);
    }
    return (MAIL_OPEN_YES);
}
