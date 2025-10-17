/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 3, 2023.
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

#include <msg.h>
#include <mymalloc.h>
#include <scan_dir.h>

/* Global library. */

#include <mail_scan_dir.h>

/* Application-specific. */

#include "qmgr.h"

/* qmgr_scan_start - start queue scan */

static void qmgr_scan_start(QMGR_SCAN *scan_info)
{
    const char *myname = "qmgr_scan_start";

    /*
     * Sanity check.
     */
    if (scan_info->handle)
	msg_panic("%s: %s queue scan in progress",
		  myname, scan_info->queue);

    /*
     * Give the poor tester a clue.
     */
    if (msg_verbose)
	msg_info("%s: %sstart %s queue scan",
		 myname,
		 scan_info->nflags & QMGR_SCAN_START ? "re" : "",
		 scan_info->queue);

    /*
     * Start or restart the scan.
     */
    scan_info->flags = scan_info->nflags;
    scan_info->nflags = 0;
    scan_info->handle = scan_dir_open(scan_info->queue);
}

/* qmgr_scan_request - request for future scan */

void    qmgr_scan_request(QMGR_SCAN *scan_info, int flags)
{

    /*
     * Apply "forget all dead destinations" requests immediately. Throttle
     * dead transports and queues at the earliest opportunity: preferably
     * during an already ongoing queue scan, otherwise the throttling will
     * have to wait until a "start scan" trigger arrives.
     * 
     * The QMGR_FLUSH_ONCE request always comes with QMGR_FLUSH_DFXP, and
     * sometimes it also comes with QMGR_SCAN_ALL. It becomes a completely
     * different story when a flush request is encoded in file permissions.
     */
    if (flags & QMGR_FLUSH_ONCE)
	qmgr_enable_all();

    /*
     * Apply "ignore time stamp" requests also towards the scan that is
     * already in progress.
     */
    if (scan_info->handle != 0 && (flags & QMGR_SCAN_ALL))
	scan_info->flags |= QMGR_SCAN_ALL;

    /*
     * Apply "override defer_transports" requests also towards the scan that
     * is already in progress.
     */
    if (scan_info->handle != 0 && (flags & QMGR_FLUSH_DFXP))
	scan_info->flags |= QMGR_FLUSH_DFXP;

    /*
     * If a scan is in progress, just record the request.
     */
    scan_info->nflags |= flags;
    if (scan_info->handle == 0 && (flags & QMGR_SCAN_START) != 0) {
	scan_info->nflags &= ~QMGR_SCAN_START;
	qmgr_scan_start(scan_info);
    }
}

/* qmgr_scan_next - look for next queue file */

char   *qmgr_scan_next(QMGR_SCAN *scan_info)
{
    char   *path = 0;

    /*
     * Restart the scan if we reach the end and a queue scan request has
     * arrived in the mean time.
     */
    if (scan_info->handle && (path = mail_scan_dir_next(scan_info->handle)) == 0) {
	scan_info->handle = scan_dir_close(scan_info->handle);
	if (msg_verbose && (scan_info->nflags & QMGR_SCAN_START) == 0)
	    msg_info("done %s queue scan", scan_info->queue);
    }
    if (!scan_info->handle && (scan_info->nflags & QMGR_SCAN_START)) {
	qmgr_scan_start(scan_info);
	path = mail_scan_dir_next(scan_info->handle);
    }
    return (path);
}

/* qmgr_scan_create - create queue scan context */

QMGR_SCAN *qmgr_scan_create(const char *queue)
{
    QMGR_SCAN *scan_info;

    scan_info = (QMGR_SCAN *) mymalloc(sizeof(*scan_info));
    scan_info->queue = mystrdup(queue);
    scan_info->flags = scan_info->nflags = 0;
    scan_info->handle = 0;
    return (scan_info);
}
