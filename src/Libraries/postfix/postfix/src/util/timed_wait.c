/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 7, 2025.
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
#include <sys/wait.h>
#include <unistd.h>
#include <signal.h>
#include <errno.h>

/* Utility library. */

#include <msg.h>
#include <posix_signals.h>
#include <timed_wait.h>

/* Application-specific. */

static int timed_wait_expired;

/* timed_wait_alarm - timeout handler */

static void timed_wait_alarm(int unused_sig)
{

    /*
     * WARNING WARNING WARNING.
     * 
     * This code runs at unpredictable moments, as a signal handler. This code
     * is here only so that we can break out of waitpid(). Don't put any code
     * here other than for setting a global flag.
     */
    timed_wait_expired = 1;
}

/* timed_waitpid - waitpid with time limit */

int     timed_waitpid(pid_t pid, WAIT_STATUS_T *statusp, int options,
		              int time_limit)
{
    const char *myname = "timed_waitpid";
    struct sigaction action;
    struct sigaction old_action;
    int     time_left;
    int     wpid;

    /*
     * Sanity checks.
     */
    if (time_limit <= 0)
	msg_panic("%s: bad time limit: %d", myname, time_limit);

    /*
     * Set up a timer.
     */
    sigemptyset(&action.sa_mask);
    action.sa_flags = 0;
    action.sa_handler = timed_wait_alarm;
    if (sigaction(SIGALRM, &action, &old_action) < 0)
	msg_fatal("%s: sigaction(SIGALRM): %m", myname);
    timed_wait_expired = 0;
    time_left = alarm(time_limit);

    /*
     * Wait for only a limited amount of time.
     */
    if ((wpid = waitpid(pid, statusp, options)) < 0 && timed_wait_expired)
	errno = ETIMEDOUT;

    /*
     * Cleanup.
     */
    alarm(0);
    if (sigaction(SIGALRM, &old_action, (struct sigaction *) 0) < 0)
	msg_fatal("%s: sigaction(SIGALRM): %m", myname);
    if (time_left)
	alarm(time_left);

    return (wpid);
}
