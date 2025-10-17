/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 22, 2025.
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
#include <signal.h>
#include <unistd.h>
#include <stdlib.h>

/* Utility library. */

#include <msg.h>
#include <iostuff.h>

/* Application-specific. */

#include <master.h>

/* master_monitor - fork off a foreground monitor process */

int     master_monitor(int time_limit)
{
    pid_t   pid;
    int     pipes[2];
    char    buf[1];

    /*
     * Sanity check.
     */
    if (time_limit <= 0)
	msg_panic("master_monitor: bad time limit: %d", time_limit);

    /*
     * Set up the plumbing for child-to-parent communication.
     */
    if (pipe(pipes) < 0)
	msg_fatal("pipe: %m");
    close_on_exec(pipes[0], CLOSE_ON_EXEC);
    close_on_exec(pipes[1], CLOSE_ON_EXEC);

    /*
     * Fork the child, and wait for it to report successful initialization.
     */
    switch (pid = fork()) {
    case -1:
	/* Error. */
	msg_fatal("fork: %m");
    case 0:
	/* Child. Initialize as daemon in the background. */
	close(pipes[0]);
	return (pipes[1]);
    default:
	/* Parent. Monitor the child in the foreground. */
	close(pipes[1]);
	switch (timed_read(pipes[0], buf, 1, time_limit, (void *) 0)) {
	default:
	    /* The child process still runs, but something is wrong. */
	    (void) kill(pid, SIGKILL);
	    /* FALLTHROUGH */
	case 0:
	    /* The child process exited prematurely. */
	    msg_fatal("daemon initialization failure");
	case 1:
	    /* The child process initialized successfully. */
	    exit(0);
	}
    }
}
